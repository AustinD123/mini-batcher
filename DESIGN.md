# DESIGN.md — mini_batcher

A minimal rebuild of NVIDIA Triton's dynamic batcher, built up in four steps
(`main.cpp` STEP 1-4 comments track the same progression as the git history).
This document describes the final (Step 4) architecture: a batcher thread and
N instance threads connected by two queues.

## Pipeline

```
LoadGenerator --> [RequestQueue] --> BatcherThread --> [BatchQueue] --> InstanceThread x N
   (producer)      mutex+cv+deque      (forms batches)   mutex+cv+deque      (runs the model)
```

- **RequestQueue**: one `Request` per client call. Producer pushes, batcher pops.
- **BatchQueue**: one `Batch` (a group of requests) per queue entry. Batcher
  pushes, instances pop.

Two queues, two independent mutex/condition_variable pairs. They are never
locked at the same time — see "Lock discipline" below.

## Threads

| Thread | Count | Waits on | Owns / writes | Reads (shared) |
|---|---|---|---|---|
| `LoadGenerator` | 1 | wall clock (`sleep_until`) | `next_id`, `RequestQueue.q` (push) | — |
| `BatcherThread` | 1 | `BatchQueue.cv` (idle slot available), then `RequestQueue.cv` (batch ready) | `RequestQueue.q` (pop), `BatchQueue.q` (push), `BatchQueue.idle_instances` (decrement) | `BatchQueue.idle_instances`, `BatchQueue.q.size()` |
| `InstanceThread` | N (`instances`) | `BatchQueue.cv` (batch available) | `BatchQueue.q` (pop), `BatchQueue.idle_instances` (increment) | `active_instances` (shared atomic, for the contention model) |

`Stats` (latencies, counters) is written by whichever thread just finished a
request or a `RunModel` call; every field is either behind `stats.mu` (the
latency vector) or a plain `std::atomic` (everything else), so no additional
coordination is needed between instances.

## The two queues

**`RequestQueue`**: `mutex`, `condition_variable`, `deque<unique_ptr<Request>>`,
`shutdown` flag. Exactly the Step 1-3 queue, unchanged — the batcher now plays
the role the single worker used to play.

**`BatchQueue`**: `mutex`, `condition_variable`, `deque<Batch>`,
`idle_instances` counter, `shutdown` flag. `idle_instances` has a single
writer in each direction: an instance thread increments it (and only it) when
it goes idle; the batcher decrements it (and only it) when it dispatches a
batch — never the instance that eventually pops that batch. This is
deliberate: it turns `idle_instances` into a simple reservation count instead
of something that needs compare-and-swap or double-checking. See the worked
example in the `BatcherThread` comment in `main.cpp` for why this can't double
count or go negative.

## Why the batcher waits for an idle instance first

`BatcherThread`'s first step blocks until `idle_instances > BatchQueue.q.size()`
— i.e. until there's at least one instance not already spoken for by a batch
sitting in the queue. Mirrors Triton's `RateLimiter::WaitForPayloadSlotAvailable`.

The alternative — form a batch the moment *anything* is queued, regardless of
whether an instance is free to run it — would let batches queue up faster than
instances can drain them, but each individual batch would still be small
(formed the instant something arrived, most of the time with nothing else
queued yet). Gating batch formation on instance availability means the batcher
only starts the clock on a new batch when it's actually going to be useful to
have one ready, which lets `RequestQueue` accumulate more arrivals in the
meantime — so batches form as *late*, and therefore as *large*, as the system
can afford under load. This is the whole point of separating "is there an
instance free" from "is there a batch ready": the batcher can afford to be
patient about (2) precisely because it's already confirmed (1).

## Why `wait_for` with a computed timeout (not a fixed poll interval)

`FormBatch` (the Step 3 rule, reused unchanged by the batcher) needs to wake
up exactly when the oldest queued request would age out — not sooner (wastes
CPU polling) and not later (blows the latency budget). A fixed poll interval
(e.g. "check every 1ms") would do both: usually wake too early for nothing,
and occasionally still miss the deadline by up to a full interval.

Instead, on every re-evaluation, `FormBatch` computes
`remaining = delay_us - age_us` from the oldest request's actual timestamp and
calls `cv.wait_for(lk, remaining)`. Whether that wait ends because of a
timeout, a new arrival notifying the condition variable, or a spurious
wakeup, the loop doesn't inspect *why* it woke up — it just re-evaluates
size/age/shutdown from scratch. That uniform handling is what makes spurious
wakeups harmless: the worst case is one extra loop iteration, never a missed
deadline or a stale decision.

## Lock discipline

`BatcherThread` never holds `RequestQueue.mu` and `BatchQueue.mu` at the same
time. `FormBatch` fully acquires and releases `rq.mu` internally and returns a
plain `Batch` by value; only after it returns does the batcher take `bq.mu` to
push the result. There is no window where both locks are held, so there is no
lock ordering to violate. `InstanceThread` only ever touches `bq.mu`, and
`LoadGenerator` only ever touches `rq.mu` — the batcher is the only thread
that touches both, and never concurrently.

## Shutdown sequence

1. `LoadGenerator` finishes its send window, sets `RequestQueue.shutdown`,
   notifies.
2. `BatcherThread`'s `FormBatch` calls keep dispatching whatever's queued
   (the delay is ignored once `rq.shutdown` is set — see Step 3's rule 3).
   Once `FormBatch` returns an empty batch (queue drained, `rq.shutdown`
   true), the batcher stops, sets `BatchQueue.shutdown`, notifies all
   instances.
3. Each `InstanceThread` keeps popping and running batches already in
   `BatchQueue.q` even after `bq.shutdown` is set (the predicate is
   `!q.empty() || shutdown`, checked before the "are we done" test) — it only
   returns once the queue is empty *and* shutdown is set. No batch, and so no
   request, is ever dropped on shutdown.

Verified directly: a run with a 10-second delay but a 2-second window still
completed 100% of requests, drained in a handful of oversized final batches
once shutdown forced immediate dispatch.

## Contention model

A plain `sleep_for(cost)` per instance would let throughput scale perfectly
with instance count — double the instances, double the throughput, forever.
Real accelerators don't behave that way: concurrent kernels on one GPU
contend for the same streaming multiprocessors and memory bandwidth, so
throughput scales sublinearly and per-call latency goes up under concurrency.

`RunModel` approximates this with one number, `CONTENTION` (default `0.7`,
CLI-overridable):

```
active = ++active_instances          // snapshot at call start
cost = base_cost * (1 + CONTENTION * (active - 1))
```

At `active == 1` (no concurrent instance), cost is unscaled. Each additional
concurrent instance adds `CONTENTION * base_cost` to every instance's cost —
including its own. `active_instances` is a single shared atomic incremented
before the sleep/spin and decremented after, so it reflects true concurrent
occupancy across all instance threads, not just this one's.

**Limitations (documented, not fixed):**
- **Linear, not physical.** Real GPU contention is workload- and
  hardware-dependent (kernel occupancy, memory-bound vs. compute-bound,
  etc.), not a single linear coefficient. `CONTENTION` is a tunable knob for
  exploring the *qualitative* effect (diminishing, even negative, returns
  from adding instances), not a calibrated model of any real accelerator.
- **Fixed at call start.** `active` is read once, at the top of `RunModel`,
  and used for the whole call's duration. If another instance starts or
  finishes mid-call, this call's cost doesn't adjust — it already committed
  to a scale factor based on the concurrency snapshot at the moment it began.
  A more faithful model would integrate concurrency continuously over the
  call's duration; this one trades that accuracy for staying simple enough
  to reason about in a learning project.
- **No queueing inside the GPU itself.** Real accelerators also serialize
  some work (e.g. memory copies); this model only scales wall-clock cost, it
  doesn't model any additional queueing beyond what `BatchQueue` already
  does.

## Triton source mapping

| mini_batcher | Triton source (triton core) |
|---|---|
| `FormBatch` (size/delay/timed-wait rule) | `DynamicBatchScheduler::GetDynamicBatch` |
| the `wait_for(remaining)` branch specifically | `DynamicBatchScheduler::DelayScheduler` |
| `BatcherThread` step 1 (wait for idle slot) | `RateLimiter::WaitForPayloadSlotAvailable` |
| `BatcherThread` step 3 (push + notify) | `RateLimiter::EnqueuePayload` |
| `InstanceThread` | `TritonBackendThread::BackendThread` |
| `MAX_QUEUE_DELAY_US` | model config `dynamic_batching.max_queue_delay_microseconds` |
| `MAX_BATCH_SIZE` | model config `dynamic_batching.preferred_batch_size` / `max_batch_size` (simplified: mini_batcher has one hard cap, not a preferred-size list) |
| `instances` (CLI) | model config `instance_group[].count` |

This is a teaching-scale simplification of Triton's actual scheduler (no
priority levels, no preferred-batch-size list, no per-model queue policies,
no ragged-batch input handling) — the mapping is to the *shape* of the real
design, not a claim of feature parity.
