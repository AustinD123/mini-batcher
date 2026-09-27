// mini_batcher.cpp — a tiny dynamic batcher, built step by step.
// Build: g++ -std=c++17 -O2 -pthread mini_batcher.cpp -o mini_batcher (add -lwinmm on Windows)
// Run:   ./mini_batcher <requests_per_second> <duration_seconds>
//
// STEP 1 (now): one worker thread, NO batching. Pop one request, run the
//               model on it, call its callback. Get this working first.
// STEP 2:       worker becomes a batcher: pop up to MAX_BATCH_SIZE at once.
// STEP 3:       add MAX_QUEUE_DELAY using cv.wait_for with a computed timeout.
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
  std::this_thread::sleep_for(cost);
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

// ---------- Latency stats (filled in by callbacks) ----------
struct Stats {
  std::mutex mu;
  std::vector<double> latencies_ms;
  std::atomic<int> completed{0};
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
    rq.cv.notify_one();
  }
  {
    std::lock_guard<std::mutex> lk(rq.mu);
    rq.shutdown = true;
  }
  rq.cv.notify_all();
}

// ---------- Worker: YOUR CODE GOES HERE ----------
void Worker(RequestQueue& rq) {
  // STEP 1 TODO:
  //   loop forever:
  //     1. lock rq.mu and wait on rq.cv until the queue is non-empty OR shutdown
  //     2. if shutdown and queue empty -> return
  //     3. pop ONE request from the front (move the unique_ptr out)
  //     4. UNLOCK before running the model (why? think about the producer)
  //     5. build a batch vector containing just that request, call RunModel
  //     6. call the request's callback with outputs[0]
  while (true) {
    std::unique_ptr<Request> req;
    {
      std::unique_lock<std::mutex> lk(rq.mu);
      rq.cv.wait(lk, [&rq] { return !rq.q.empty() || rq.shutdown; });
      if (rq.shutdown && rq.q.empty()) return;
      req = std::move(rq.q.front());
      rq.q.pop_front();
    }
    std::vector<std::unique_ptr<Request>> batch;
    batch.push_back(std::move(req));
    auto outputs = RunModel(batch);
    batch[0]->callback(batch[0]->id, outputs[0]);
  }

}

int main(int argc, char** argv) {
#ifdef _WIN32
  WinTimerResolution timer_resolution;
#endif
  int rps = argc > 1 ? std::atoi(argv[1]) : 50;
  int seconds = argc > 2 ? std::atoi(argv[2]) : 3;

  RequestQueue rq;
  Stats stats;

  std::thread worker(Worker, std::ref(rq));
  std::thread producer(LoadGenerator, std::ref(rq), std::ref(stats), rps, seconds);

  producer.join();
  worker.join();

  // Report
  std::lock_guard<std::mutex> lk(stats.mu);
  auto& v = stats.latencies_ms;
  if (v.empty()) { std::printf("no requests completed\n"); return 0; }
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  std::printf("rps=%d completed=%zu  avg=%.1fms  p50=%.1fms  p99=%.1fms  max=%.1fms\n",
              rps, v.size(), sum / v.size(), v[v.size() / 2],
              v[std::min(v.size() - 1, v.size() * 99 / 100)], v.back());
}
