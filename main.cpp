// mini_batcher.cpp — a tiny dynamic batcher, built step by step.
// Build: g++ -std=c++17 -O2 -pthread mini_batcher.cpp -o mini_batcher (add -lwinmm on Windows)
// Run:   ./mini_batcher <rps> <seconds> [instances=1] [delay_us=5000] [contention=0.7]
//
// STEP 1: one worker thread, NO batching. Pop one request, run the model
//         on it, call its callback.
// STEP 2: worker becomes a batcher: pop up to MAX_BATCH_SIZE at once.
// STEP 3: add MAX_QUEUE_DELAY_US using cv.wait_for with a computed timeout,
//         so a batch either fills up or ages out.
// STEP 4 (now): split into a batcher thread + N instance threads with a
//               second (batch) queue; the batcher waits for an idle
//               instance before forming each batch. See DESIGN.md.

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

// Largest batch an instance will run in one RunModel() call.
constexpr size_t MAX_BATCH_SIZE = 8;

// Default max time a request waits in queue before its (possibly
// under-full) batch is dispatched anyway. Overridable from the CLI.
// mirrors the model config's dynamic_batching.max_queue_delay_microseconds.
constexpr long kDefaultMaxQueueDelayUs = 5000;

// Default shared-GPU contention factor. Overridable from the CLI.
constexpr double kDefaultContention = 0.7;

// ---------- Request: what a "client" sends ----------
struct Request {
  int id;
  std::vector<float> input;
  Clock::time_point enqueue_time;
  std::function<void(int id, float output)> callback;  // the "return address"
};

using Batch = std::vector<std::unique_ptr<Request>>;

// ---------- Fake model: bigger batches cost almost the same, and
// concurrent instances contend for the same (simulated) GPU ----------
// Base cost = 10 ms + 0.5 ms per item. Output = sum of each input.
std::vector<float> RunModel(const Batch& batch, std::atomic<int>& active_instances,
                             double contention) {
  int active = ++active_instances;
  auto base_cost = std::chrono::microseconds(10000 + 500 * batch.size());

  // Shared-GPU contention model: a plain sleep would let N instances scale
  // throughput perfectly, which real accelerators don't -- concurrent
  // kernels contend for the same SMs/memory bandwidth. Approximate that by
  // scaling every instance's cost by how many instances are active right
  // now: cost *= 1 + CONTENTION * (active - 1). See DESIGN.md for what
  // this model gets wrong (it's a linear stand-in for a nonlinear effect,
  // and it's fixed at call-start rather than tracking concurrency that
  // changes mid-call).
  double scale = 1.0 + contention * (active - 1);
  auto cost =
      std::chrono::microseconds(static_cast<long long>(base_cost.count() * scale));
  auto deadline = Clock::now() + cost;

  // Hybrid wait: this host's sleep_for() itself is quantized to ~10-20ms
  // even with timeBeginPeriod(1) (measured directly), so a small margin
  // isn't enough slack. 12ms was empirically the smallest margin that
  // landed on the exact deadline consistently; sleep for the coarse
  // majority, then spin against the clock for the rest.
  constexpr auto kSpinMargin = std::chrono::milliseconds(12);
  if (cost > kSpinMargin) {
    std::this_thread::sleep_for(cost - kSpinMargin);
  }
  while (Clock::now() < deadline) {
  }

  --active_instances;

  std::vector<float> outputs;
  for (const auto& r : batch) {
    float sum = 0;
    for (float x : r->input) sum += x;
    outputs.push_back(sum);  // position i in outputs belongs to batch[i]
  }
  return outputs;
}

// ---------- Request queue (shared between producer and batcher) ----------
struct RequestQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::unique_ptr<Request>> q;
  bool shutdown = false;
};

// ---------- Forms one batch from the shared request queue ----------
// mirrors DynamicBatchScheduler::GetDynamicBatch (triton core): dispatch
// once the batch is full, the oldest request has aged out, or we're
// shutting down and can't afford to wait any longer. Re-evaluated from
// scratch on every wakeup -- a real notify, a spurious wakeup, and a
// timeout all funnel through the same check. Returns an empty batch only
// when the producer is done AND the queue is fully drained; callers should
// treat that as "no more work, ever."
Batch FormBatch(RequestQueue& rq, long delay_us) {
  Batch batch;
  std::unique_lock<std::mutex> lk(rq.mu);
  rq.cv.wait(lk, [&rq] { return !rq.q.empty() || rq.shutdown; });
  if (rq.shutdown && rq.q.empty()) return batch;

  while (batch.empty()) {
    size_t n = rq.q.size();
    auto age_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      Clock::now() - rq.q.front()->enqueue_time)
                      .count();

    // 1. full batch, 2. oldest request aged out, or 3. shutting down and
    // can't wait any longer -- dispatch what's queued right now.
    bool dispatch_now = n >= MAX_BATCH_SIZE || age_us >= delay_us || rq.shutdown;

    if (dispatch_now) {
      size_t take = std::min(n, MAX_BATCH_SIZE);
      for (size_t i = 0; i < take; ++i) {
        batch.push_back(std::move(rq.q.front()));
        rq.q.pop_front();
      }
    } else {
      // mirrors DynamicBatchScheduler::DelayScheduler: sleep only until the
      // oldest request would age out, not the full delay again.
      auto remaining =
          std::chrono::microseconds(delay_us) - std::chrono::microseconds(age_us);
      rq.cv.wait_for(lk, remaining);
      // Don't inspect the wait_for() result or assume why we woke up --
      // loop back and re-check size/age/shutdown from scratch.
    }
  }
  return batch;
}

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
void LoadGenerator(RequestQueue& rq, Stats& stats, std::vector<std::atomic<int>>& callback_counts,
                    int rps, int seconds) {
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
    req->callback = [&stats, &callback_counts, req_start,
                      id = req->id](int /*id*/, float /*output*/) {
      double ms = std::chrono::duration<double, std::milli>(Clock::now() - req_start).count();
      std::lock_guard<std::mutex> lk(stats.mu);
      stats.latencies_ms.push_back(ms);
      stats.completed++;
      callback_counts[id]++;  // correctness check: must land on exactly 1
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

// ---------- Batch queue (shared between the batcher and instance threads)
struct BatchQueue {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Batch> q;
  // Only ever incremented by an instance thread (when it goes idle) and
  // decremented by the batcher thread (when it dispatches a batch) -- see
  // BatcherThread for why that single-writer-per-direction split is safe.
  int idle_instances = 0;
  bool shutdown = false;
};

// ---------- Instance thread: runs the model, one batch at a time ----------
// mirrors TritonBackendThread::BackendThread.
void InstanceThread(BatchQueue& bq, Stats& stats, std::atomic<int>& active_instances,
                     double contention) {
  while (true) {
    Batch batch;
    {
      std::unique_lock<std::mutex> lk(bq.mu);
      bq.idle_instances++;
      bq.cv.notify_all();  // wake the batcher: a slot may have opened up
      bq.cv.wait(lk, [&bq] { return !bq.q.empty() || bq.shutdown; });
      if (bq.q.empty() && bq.shutdown) return;
      batch = std::move(bq.q.front());
      bq.q.pop_front();
    }  // unlock before running the model, same reasoning as every step before this

    auto model_start = Clock::now();
    auto outputs = RunModel(batch, active_instances, contention);
    auto model_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - model_start)
                         .count();
    stats.runmodel_calls++;
    stats.batch_items += batch.size();
    stats.model_busy_us += model_us;

    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i]->callback(batch[i]->id, outputs[i]);  // outputs[i] belongs to batch[i]
    }
  }
}

// ---------- Batcher thread: forms batches, hands them to idle instances ----------
// mirrors BatcherThread. Never holds rq.mu and bq.mu at the same time:
// FormBatch() fully releases rq.mu before returning, and only then do we
// take bq.mu to push the result -- there's never a moment both are held,
// so there's no lock order to get wrong.
void BatcherThread(RequestQueue& rq, BatchQueue& bq, long delay_us) {
  while (true) {
    // 1. mirrors RateLimiter::WaitForPayloadSlotAvailable: don't start
    // forming a batch until an instance is actually free to run it. This
    // is what makes batches form as LATE as possible under load, so
    // they're as LARGE as possible by the time they're dispatched.
    {
      std::unique_lock<std::mutex> lk(bq.mu);
      bq.cv.wait(lk, [&bq] { return bq.idle_instances > (int)bq.q.size(); });
    }

    // 2. mirrors DynamicBatchScheduler::GetDynamicBatch (the Step 3 rule).
    Batch batch = FormBatch(rq, delay_us);
    if (batch.empty()) break;  // producer done, RequestQueue fully drained

    // 3. mirrors RateLimiter::EnqueuePayload. idle_instances is decremented
    // here -- at dispatch time -- rather than by whichever instance
    // eventually pops this batch, so a second batch can never be formed
    // for a slot that's already spoken for.
    {
      std::lock_guard<std::mutex> lk(bq.mu);
      bq.q.push_back(std::move(batch));
      bq.idle_instances--;
    }
    bq.cv.notify_one();
  }

  // Producer is done and RequestQueue is empty: no more batches will ever
  // be formed. Tell every instance so they can drain whatever's still
  // queued in bq.q and exit -- no request may be dropped.
  {
    std::lock_guard<std::mutex> lk(bq.mu);
    bq.shutdown = true;
  }
  bq.cv.notify_all();
}

int main(int argc, char** argv) {
#ifdef _WIN32
  WinTimerResolution timer_resolution;
#endif
  int rps = argc > 1 ? std::atoi(argv[1]) : 50;
  int seconds = argc > 2 ? std::atoi(argv[2]) : 3;
  int instances = argc > 3 ? std::atoi(argv[3]) : 1;
  long delay_us = argc > 4 ? std::atol(argv[4]) : kDefaultMaxQueueDelayUs;
  double contention = argc > 5 ? std::atof(argv[5]) : kDefaultContention;

  RequestQueue rq;
  BatchQueue bq;
  Stats stats;
  std::atomic<int> active_instances{0};

  // Sized generously so every request id (0..sent-1) has a slot; checked
  // against `sent` after the run rather than trusted blindly.
  std::vector<std::atomic<int>> callback_counts((size_t)rps * (seconds + 1) + 4096);
  for (auto& c : callback_counts) c = 0;

  auto wall_start = Clock::now();

  std::vector<std::thread> instance_threads;
  for (int i = 0; i < instances; ++i) {
    instance_threads.emplace_back(InstanceThread, std::ref(bq), std::ref(stats),
                                   std::ref(active_instances), contention);
  }
  std::thread batcher(BatcherThread, std::ref(rq), std::ref(bq), delay_us);
  std::thread producer(LoadGenerator, std::ref(rq), std::ref(stats), std::ref(callback_counts),
                        rps, seconds);

  producer.join();
  batcher.join();
  for (auto& t : instance_threads) t.join();
  auto wall_us = std::chrono::duration<double, std::micro>(Clock::now() - wall_start).count();

  if (stats.completed.load() != stats.sent.load()) {
    std::fprintf(stderr,
                 "!!! ERROR: completed (%d) != sent (%d) -- requests were dropped or "
                 "double-counted !!!\n",
                 stats.completed.load(), stats.sent.load());
  }
  int bad_ids = 0;
  for (int i = 0; i < stats.sent.load(); ++i) {
    if (callback_counts[i].load() != 1) bad_ids++;
  }
  if (bad_ids > 0) {
    std::fprintf(stderr,
                 "!!! ERROR: %d request id(s) had a callback fire count != 1 (dropped or "
                 "double-fired) !!!\n",
                 bad_ids);
  }

  // Report
  std::lock_guard<std::mutex> lk(stats.mu);
  auto& v = stats.latencies_ms;
  double avg_batch = stats.runmodel_calls.load() > 0
                          ? (double)stats.batch_items.load() / stats.runmodel_calls.load()
                          : 0.0;
  // busy_pct is averaged per instance: total time any instance spent inside
  // RunModel, divided by (wall time * instance count). 100% means every
  // instance was busy for the whole run.
  double busy_pct =
      wall_us > 0 ? 100.0 * stats.model_busy_us.load() / (wall_us * instances) : 0.0;
  if (v.empty()) {
    std::printf("rps=%d sent=%d completed=0  no requests completed\n", rps, stats.sent.load());
    return 0;
  }
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  std::printf(
      "rps=%d completed=%zu sent=%d  avg=%.1fms  p50=%.1fms  p99=%.1fms  max=%.1fms  "
      "runmodel_calls=%lld  avg_batch=%.2f  busy_pct=%.1f  instances=%d delay_us=%ld "
      "contention=%.2f\n",
      rps, v.size(), stats.sent.load(), sum / v.size(), v[v.size() / 2],
      v[std::min(v.size() - 1, v.size() * 99 / 100)], v.back(), stats.runmodel_calls.load(),
      avg_batch, busy_pct, instances, delay_us, contention);
}
