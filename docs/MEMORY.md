# The memory hazard

Read this before you start a model. On this machine an over-committed vLLM
server does not run slowly and it does not get killed — it takes the whole box
down, and getting it back may mean walking over to the power button.

## GPU memory is host memory

The GB10 in a DGX Spark has no separate framebuffer. CPU and GPU address one
pool of memory, and the numbers agree exactly:

```
/proc/meminfo MemTotal      127600592 kB  = 130,663,006,208 bytes = 121.69 GiB
torch.cuda.mem_get_info()   130.66e9 bytes                        = 121.69 GiB
```

That is not a coincidence or a rounding artefact; it is the same memory counted
twice. Every consequence below follows from it.

`nvidia-smi` reflects this by refusing to answer. `memory.total`, `memory.used`,
`memory.free`, `power.limit`, `clocks.current.memory` and `fan.speed` all return
`[N/A]` on this part, because there is no discrete framebuffer for them to
describe. The one memory query that works is
`--query-compute-apps=pid,used_gpu_memory`, which reports real per-process
bytes. The dashboard therefore reads host memory from `/proc/meminfo` and
per-model memory from `--query-compute-apps`, and shows nothing where the driver
has nothing to say.

## What `--gpu-memory-utilization` actually claims

vLLM sizes its allocation as `utilization × total device memory`. Total device
memory here is MemTotal. So:

```
--gpu-memory-utilization 0.52   ->   0.52 × 121.69 GiB = 63.3 GiB of host RAM
--gpu-memory-utilization 0.16   ->   0.16 × 121.69 GiB = 19.5 GiB of host RAM
--gpu-memory-utilization 0.80   ->   0.80 × 121.69 GiB = 97.4 GiB of host RAM
```

The fractions of co-resident engines add up. Two servers at 0.52 and 0.16 have
claimed 0.68 of the machine between them, and only 0.32 — about 39 GiB — is left
for the kernel, page cache, the checkpoint reads of the *next* model to load,
torch.compile, and everything else you are running.

Nothing enforces the sum. Each engine independently believes it may take its
fraction of 121.69 GiB, so two servers configured at 0.6 each will both start,
both succeed, and meet somewhere in the middle of a machine that has no memory
left.

## Measured values on this box

Three data points, all from this hardware:

| configuration | sum of utils | what happened |
|---|---|---|
| 27B NVFP4 at 0.52 beside an 8B embedder at 0.16 | 0.68 | stable; what this box runs |
| the 27B alone at 0.57 | 0.57 | stable |
| a single engine at 0.80 | 0.80 | hard-locked the machine |

The 0.80 case is the important one, and it explains the reserve the dashboard
insists on. It did not get OOM-killed. This host has 16 GiB of swap, so when
allocation outruns physical memory the kernel starts swapping rather than
invoking the OOM killer, and a machine whose page cache and anonymous memory are
both on swap stops responding to anything, including the thing that would have
saved it.

The 0.68 configuration also shows why a smaller reserve is not enough. During
the 27B's weight load, with only 0.68 committed, vLLM logged
`Checkpoint size: 21.81 GiB. Available RAM: 69.18 GiB` and MemFree bottomed out
around 4.8 GiB. The utilisation reservation is not the only claim on memory: the
checkpoint read goes through page cache, and torch.compile and CUDA graph
capture want their own working set on top. A reserve sized to "the OS needs a
few gigabytes" would have accepted the 0.80 launch that locks the box.

## The arithmetic the dashboard uses

`app/safety.py` runs this before every launch. It is deliberately conservative
and it counts containers it did not start.

```
total      = MemTotal                                    (121.69 GiB)
committed  = Σ util of every RUNNING vLLM container × total
measured   = Σ per-process bytes from --query-compute-apps
occupied   = max(committed, measured)
reserve    = LLMD_MEM_RESERVE_GIB                        (32 GiB by default)
ceiling    = 1 − reserve/total                           (0.737)
```

`committed` comes from parsing `--gpu-memory-utilization` out of each running
container's recorded command line, which is why a `vllm serve` you launched by
hand from a shell script counts exactly like a server the dashboard created. A
serve command with **no** utilisation flag is not free: vLLM applies its own
default, read from the generated schema and currently 0.92, and the budget
charges it that.

`measured` exists because the util sum only knows about vLLM. A Heretic run, a
fine-tuning job or anything else holding GPU memory shows up in the per-process
figures and nowhere else, so whichever number is larger is the one used.

A launch at `requested` is then judged:

* **blocked** if `occupied + requested×total > total − reserve` — the launch
  would eat the reserve.
* **blocked** if `requested×total > MemAvailable` — the memory is not there
  right now, whatever the budget says.
* **warned** if `requested×total > MemFree` — it fits in MemAvailable but not in
  MemFree, so the difference is reclaimable page cache. vLLM's own pre-flight
  compares against free memory and may refuse with "Free memory on device is
  less than desired GPU memory utilization" even though the launch is legal.
* **warned** if the total would leave less than `LLMD_MEM_WARN_RESERVE_GIB`
  (38 GiB) free.
* **allowed** otherwise.

The verdict names the largest utilisation that would still fit — that is
`ceiling − occupied/total` — and the Serve tab prints it under the message.
With the two servers above resident, that number is 0.057: about 6.9 GiB, which
is room for a small model and nothing else. That is what a full box looks like.

Jobs that load a model outside vLLM take no utilisation fraction at all. Heretic
loads with `device_map="auto"` and, on a unified-memory part, sees all 121.69
GiB as VRAM. Those runs are pre-flighted differently: the dashboard estimates
resident size from the parameter count (and, for Heretic, adds the merge spike,
which is the real peak) and compares it against MemAvailable. It is an estimate,
not a reservation, and the warning says so.

## Why `--max-num-seqs` is the safety valve

The reservation is made once, at startup, and vLLM sizes it by profiling a peak
forward pass at `--max-num-seqs` sequences and `--max-num-batched-tokens`
tokens, then giving whatever is left of the fraction to the KV cache. Two things
follow.

A high `--max-num-seqs` spends the budget on activation headroom rather than KV
cache, and it raises the transient peak the engine can reach at run time — the
moment where a unified-memory host has nothing to fall back on. The measured-safe
27B configuration runs `--max-num-seqs 16`; 8 is the conservative end of its
range.

And it is the *only* lever that bounds in-flight work. `--max-num-seqs` is an
admission cap, not a request limit: requests beyond it queue in the scheduler
instead of being rejected, so clients see latency, not errors. Rate-limiting in
front of the server does not help, because the engine batches whatever has
arrived; capping concurrency inside the engine is what keeps the activation
peak where the profiler measured it.

`--kv-cache-dtype fp8` and a realistic `--max-model-len` are the other two
levers worth reaching for before you raise the utilisation fraction.

## Why `docker run --memory` cannot save you

It bounds the wrong number. Measured on this host, at a moment when
`nvidia-smi` reported the vllm-qwen engine process holding 62 GiB:

```
docker stats --no-stream   ->   vllm-qwen   4.548GiB / 121.7GiB   3.74%
```

The memory cgroup is not broken — a container given `--memory 32m` and told to
allocate was OOM-killed with exit 137 and `OOMKilled: true`. It is specifically
that CUDA's unified allocations are not charged to the cgroup that owns the
process. A `--memory` limit on a vLLM container will therefore either do nothing
or kill the container for the few gigabytes of ordinary host allocations it also
makes, while the 60 GiB that actually threatens the machine passes straight
through.

There is no enforcement backstop on this hardware. Admission control before
launch is the only real protection, which is why the dashboard blocks rather
than warns, and why forcing past a block is a deliberate, separate action.

## What the watchdog does

`app/memguard.py` runs inside the dashboard (and standalone via
`scripts/memguard.sh`). Every two seconds it reads MemAvailable. Below
`LLMD_MEMGUARD_THRESHOLD_MIB` — 10240 MiB by default — it picks the running
container with the largest declared utilisation, `docker kill`s it, and sets its
restart policy to `no` so it cannot come back and re-reserve the same memory.
Then it waits 15 seconds before considering another victim, because freed memory
takes a moment to show up in MemAvailable.

It only ever kills a container whose command line is a vLLM `serve`. It will not
touch the dashboard, a fine-tuning run, or a container it cannot identify. Kills
are recorded (the last 50) and pushed to the Overview page, so a server that
vanished has a visible reason.

It watches MemAvailable and not MemFree on purpose. MemFree collapses during
every normal weight load — 4.8 GiB while a 21.81 GiB checkpoint streams through
page cache — while MemAvailable, which counts reclaimable cache, held near 30
GiB through the same load. A MemFree-driven watchdog would kill a healthy server
every time you started one.

Understand what it is for. By the time MemAvailable is falling through 10 GiB,
the allocation that caused it has already been made; killing a container is
damage control, not prevention. A spike faster than the two-second poll will
beat it. Treat it as the thing that keeps a bad launch from costing you the
uptime of everything else on the box, and treat the launch check as the thing
that keeps the bad launch from happening.

## If it happens anyway

Symptoms: the UI stops updating, ssh takes tens of seconds per keystroke, and
`SwapFree` is falling. That last one is the earliest signal — swap dropping by
more than about a gigabyte over half a minute means the kernel is thrashing.

While you can still get a command in, kill the biggest engine directly:

```
docker ps --format '{{.Names}}'
docker kill <name>
```

There is no sudo on this box, so dropping caches or changing swappiness is not
available. If the machine has stopped scheduling your shell, the reset button is
the remaining option — which is the entire reason the launch check refuses
instead of asking.
