// mini_batcher.cpp — a tiny dynamic batcher, built step by step.
// Build: g++ -std=c++17 -O2 -pthread mini_batcher.cpp -o mini_batcher (add -lwinmm on Windows)
// Run:   ./mini_batcher <requests_per_second> <duration_seconds> [delay_us]
//
// STEP 1: one worker thread, NO batching. Pop one request, run the model
//         on it, call its callback.
// STEP 2: worker becomes a batcher: pop up to MAX_BATCH_SIZE at once.
// STEP 3 (now): add MAX_QUEUE_DELAY_US using cv.wait_for with a computed
//               timeout, so a batch either fills up or ages out.
// STEP 4:       split into a batcher thread + N instance threads with a
//               second (batch) queue; batcher waits for an idle instance.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
// RAII wrapper: Windows' default timer resolution is ~15.6ms, which is
// coarse enough to distort both sleep_until() and our latency measurements.
// timeBeginPeriod(1) requests 1ms resolution for the process; the matching
// timeEndPeriod(1) on destruction restores the system default, and using
// RAII means it fires on every exit path (including the early return below).
struct WinTimerResolution {
  WinTimerResolution() { timeBeginPeriod(1); }
  ~WinTimerResolution() { timeEndPeriod(1); }
};
#endif

using Clock = std::chrono::steady_clock;

// Largest batch the worker will assemble in one RunModel() call.
constexpr size_t MAX_BATCH_SIZE = 8;

// Default max time a request waits in queue before its (possibly
// under-full) batch is dispatched anyway. Overridable from the CLI.
// mirrors the model config's dynamic_batching.max_queue_delay_microseconds.
constexpr long kDefaultMaxQueueDelayUs = 5000;

// ---------- Request: what a "client" sends ----------
struct Request {
  int id;
  std::vector<float> input;
  Clock::time_point enqueue_time;
  std::function<void(int id, float output)> callback;  // the "return address"
};

// ---------- Fake model: bigger batches cost almost the same ----------
// Cost = 10 ms base + 0.5 ms per item. Output = sum of each input.
std::vector<float> RunModel(const std::vector<std::unique_ptr<Request>>& batch) {
  auto cost = std::chrono::microseconds(10000 + 500 * batch.size());
  auto deadline = Clock::now() + cost;

  // Hybrid wait: this host's sleep_for() itself is quantized to ~10-20ms
  // even with timeBeginPeriod(1) (measured directly), so a small margin
  // isn't enough slack — sleep_for(cost - 2ms) still overshoots the
  // deadline on ~80% of calls here. 12ms was empirically the smallest
  // margin that landed on the exact deadline consistently (10/10 trials);
  // sleep for the coarse majority, then spin against the clock for the rest.
  constexpr auto kSpinMargin = std::chrono::milliseconds(12);
  if (cost > kSpinMargin) {
    std::this_thread::sleep_for(cost - kSpinMargin);
  }
  while (Clock::now() < deadline) {
  }

  std::vector<float> outputs;
  for (const auto& r : batch) {
    float sum = 0;
    for (float x : r->input) sum += x;
    outputs.push_back(sum);  // position i in outputs belongs to batch[i]
  }
  return outputs;
}

// ---------- Request queue (shared between producer and worker) ----------
struct RequestQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::unique_ptr<Request>> q;
  bool shutdown = false;
};

// ---------- Latency + throughput stats ----------
struct Stats {
  std::mutex mu;
  std::vector<double> latencies_ms;
  std::atomic<int> completed{0};
  std::atomic<int> sent{0};               // requests the producer actually enqueued
  std::atomic<long long> runmodel_calls{0};
  std::atomic<long long> batch_items{0};  // sum of batch sizes, for the average
  std::atomic<long long> model_busy_us{0};  // sum of time spent inside RunModel
};

// ---------- Load generator: plays the role of the HTTP thread ----------
void LoadGenerator(RequestQueue& rq, Stats& stats, int rps, int seconds) {
  auto interval = std::chrono::microseconds(1000000 / rps);
  auto start = Clock::now();
  auto end = start + std::chrono::seconds(seconds);
  int next_id = 0;
  while (true) {
    // Absolute schedule: target time for request i is start + i * interval,
    // not "now + interval". This keeps the send rate accurate even if a
    // single sleep overshoots — sleep_for would let each overshoot compound
    // (drift), while recomputing from `start` each time cannot drift.
    auto target = start + next_id * interval;
    if (target >= end) break;
    if (Clock::now() < target) {
      std::this_thread::sleep_until(target);
    }
    // else: already behind schedule — send immediately to catch up instead
    // of pushing every subsequent request later still.

    auto req = std::make_unique<Request>();
    req->id = next_id++;
    req->input = {1.0f, 2.0f, 3.0f};
    req->enqueue_time = Clock::now();
    auto req_start = req->enqueue_time;
    req->callback = [&stats, req_start](int /*id*/, float /*output*/) {
      double ms = std::chrono::duration<double, std::milli>(Clock::now() - req_start).count();
      std::lock_guard<std::mutex> lk(stats.mu);
      stats.latencies_ms.push_back(ms);
      stats.completed++;
    };
    {
      std::lock_guard<std::mutex> lk(rq.mu);
      rq.q.push_back(std::move(req));
    }
    stats.sent++;
    rq.cv.notify_one();
  }
  {
    std::lock_guard<std::mutex> lk(rq.mu);
    rq.shutdown = true;
  }
  rq.cv.notify_all();
}

// ---------- Worker: single-thread batcher with a max queue delay ----------
void Worker(RequestQueue& rq, Stats& stats, long delay_us) {
  while (true) {
    std::vector<std::unique_ptr<Request>> batch;
    {
      std::unique_lock<std::mutex> lk(rq.mu);   // lock acquired ONCE here
      rq.cv.wait(lk, [&rq] { return !rq.q.empty() || rq.shutdown; });
      if (rq.shutdown && rq.q.empty()) return;

      // mirrors DynamicBatchScheduler::GetDynamicBatch (triton core): decide
      // whether to dispatch what's queued now, or wait longer for the batch
      // to fill up further. Re-evaluated from scratch on every wakeup —
      // real notify, spurious wakeup, and timeout all funnel through here.
      while (batch.empty()) {
        size_t n = rq.q.size();
        auto age_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          Clock::now() - rq.q.front()->enqueue_time)
                          .count();

        // 1. full batch, 2. oldest request aged out, or 3. shutting down
        // and can't afford to wait any longer — dispatch what we have.
        bool dispatch_now = n >= MAX_BATCH_SIZE || age_us >= delay_us || rq.shutdown;

        if (dispatch_now) {
          size_t take = std::min(n, MAX_BATCH_SIZE);
          for (size_t i = 0; i < take; ++i) {
            batch.push_back(std::move(rq.q.front()));
            rq.q.pop_front();
          }
        } else {
          // mirrors DynamicBatchScheduler::DelayScheduler: sleep only until
          // the oldest request would age out, not the full delay again.
          auto remaining = std::chrono::microseconds(delay_us) -
                            std::chrono::microseconds(age_us);
          rq.cv.wait_for(lk, remaining);
          // Don't inspect the wait_for result or assume why we woke up —
          // loop back to step 1 and re-check size/age/shutdown from scratch.
        }
      }
    }                                            // lock released ONCE here

    auto model_start = Clock::now();
    auto outputs = RunModel(batch);              // no lock held: producer keeps enqueuing
    auto model_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - model_start)
                         .count();
    stats.runmodel_calls++;
    stats.batch_items += batch.size();
    stats.model_busy_us += model_us;

    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i]->callback(batch[i]->id, outputs[i]);   // outputs[i] belongs to batch[i]
    }
  }
}

int main(int argc, char** argv) {
#ifdef _WIN32
  WinTimerResolution timer_resolution;
#endif
  int rps = argc > 1 ? std::atoi(argv[1]) : 50;
  int seconds = argc > 2 ? std::atoi(argv[2]) : 3;
  long delay_us = argc > 3 ? std::atol(argv[3]) : kDefaultMaxQueueDelayUs;

  RequestQueue rq;
  Stats stats;

  auto wall_start = Clock::now();
  std::thread worker(Worker, std::ref(rq), std::ref(stats), delay_us);
  std::thread producer(LoadGenerator, std::ref(rq), std::ref(stats), rps, seconds);

  producer.join();
  worker.join();
  auto wall_us = std::chrono::duration<double, std::micro>(Clock::now() - wall_start).count();

  if (stats.completed.load() != stats.sent.load()) {
    std::fprintf(stderr, "!!! ERROR: completed (%d) != sent (%d) — requests were dropped or duplicated !!!\n",
                 stats.completed.load(), stats.sent.load());
  }

  // Report
  std::lock_guard<std::mutex> lk(stats.mu);
  auto& v = stats.latencies_ms;
  double avg_batch = stats.runmodel_calls.load() > 0
                          ? (double)stats.batch_items.load() / stats.runmodel_calls.load()
                          : 0.0;
  double busy_pct = wall_us > 0 ? 100.0 * stats.model_busy_us.load() / wall_us : 0.0;
  if (v.empty()) {
    std::printf("rps=%d sent=%d completed=0  no requests completed\n", rps, stats.sent.load());
    return 0;
  }
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  std::printf("rps=%d completed=%zu sent=%d  avg=%.1fms  p50=%.1fms  p99=%.1fms  max=%.1fms  "
              "runmodel_calls=%lld  avg_batch=%.2f  busy_pct=%.1f\n",
              rps, v.size(), stats.sent.load(), sum / v.size(), v[v.size() / 2],
              v[std::min(v.size() - 1, v.size() * 99 / 100)], v.back(),
              stats.runmodel_calls.load(), avg_batch, busy_pct);
}
