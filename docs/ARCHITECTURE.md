# Architecture

The dashboard is one Python process that talks to the docker daemon, the
`/proc` filesystem and the HuggingFace Hub, and serves a browser UI that has no
build step. There is no queue, no broker, no worker pool and no state outside
one SQLite file: everything long-running is a detached container, and the
process's job is to launch, watch and explain them.

## Process topology

```
browser ──HTTP/SSE──> uvicorn (single worker, host process)
                          │
                          ├── /proc/meminfo, /proc/loadavg      host memory, load
                          ├── nvidia-smi                        per-process GPU bytes
                          ├── docker CLI                        containers, images, logs
                          ├── HuggingFace Hub over httpx        search and sizing
                          └── vLLM servers over httpx           /health, /metrics, /v1/*
```

Three constraints are baked in.

**It runs on the host, not in a container.** NVML and `nvidia-smi` inside a
container see only their own PID namespace — asked for GPU processes from inside
a vLLM container, they report that container's single PID — so a containerised
dashboard could not see how much memory the *other* models are holding. It also
needs `/proc/<pid>/cgroup` on host PIDs to attribute GPU memory to containers.

It is installed as a systemd *user* unit (`deploy/llm-dashboard.service.in`)
for the same reason: a system unit would run as root, and root is exactly what
this process should not be.

**One uvicorn worker, no `--reload`.** The telemetry poller, the log followers
and the memory watchdog are in-process state. A second worker would run a second
copy of each, and the reloader forks in a way that breaks the child-process
bookkeeping.

**The dashboard never owns a container's lifetime.** Everything is launched
detached, so restarting the web process cannot interrupt a thirty-minute model
load. On startup `jobs.manager.reconcile()` looks for containers belonging to
jobs that were running when the process died and reattaches to their logs with
`docker logs --since <cursor>`.

## Modules

Bottom to top; each layer only calls downwards.

| module | responsibility |
|---|---|
| `app/config.py` | every tunable, all overridable through `LLMD_*` environment variables. Nothing else in the host process reads the environment. |
| `app/db.py` | SQLite. The schema is one `executescript` of `CREATE TABLE IF NOT EXISTS`, run at startup. Synchronous by design; callers push it onto a thread. |
| `app/docker_ctl.py` | the only place that shells out to `docker`. Argv is built as a list and never goes through a shell. Returns typed `ContainerState`, streams logs and builds. |
| `app/telemetry.py` | `/proc/meminfo`, `/proc/loadavg`, `nvidia-smi`. Knows which fields a unified part reports as `[N/A]` and omits them rather than showing zeros. |
| `app/accel.py` | which memory `--gpu-memory-utilization` is a fraction of on a given machine: host RAM on a unified part, the framebuffer on a discrete GPU. Detects it per node and fails toward unified, because guessing discrete on a unified box is what freezes it. |
| `app/events.py` | in-process pub/sub with bounded queues, one topic per stream. Slow browser tabs drop old frames instead of growing the producer. |
| `app/safety.py` | the launch budget. Surveys every running engine container — either engine, managed or not — and returns a `Verdict` the API and UI both render. Denominated in bytes throughout; a utilisation fraction is one engine's way of stating them. |
| `app/memguard.py` | the runtime watchdog. Polls MemAvailable and kills the largest resident engine below the threshold — the whole engine, across nodes, since killing one rank of a pooled one aborts it while the rest keep their memory. Its invariant is *kill only what we understand well enough to know what dies with it*, not *kill only vLLM*. Warns instead of killing where host pressure is not the engines' doing. |
| `app/nodes.py`, `app/cluster.py`, `app/sync.py` | the cluster: registering peers reached over `docker -H ssh://…`, detecting each node's fabric interface and memory, copying a model to a node that lacks it, and building the per-rank commands for one engine split across machines. |
| `app/jobs.py` | one job = one detached container + a log file + a progress dict. Per-kind line parsers, reattach on restart, cancel. |
| `app/engines/` | the engine seam. One object per inference engine answering the argv, the image, the container kind, the environment, the metric names, how a container is recognised and what it is estimated to cost. `engines/vllm.py` is a pure adapter over `vllm_spec` and functions moved verbatim out of `safety`, so the first engine's behaviour is unchanged by the second's existence. |
| `app/vllm_spec.py`, `app/llamacpp_spec.py`, `app/sampling_spec.py` | turn generated JSON schemas into a form model and back into argv or a request body. The two engine specs share a document shape and therefore one renderer. |
| `app/gguf.py` | a bounded reader for a .gguf's own metadata header. A GGUF has no config.json — its architecture, layer count and head geometry are inside the weights file — and llama.cpp's footprint is arithmetic over exactly those. |
| `app/servers.py`, `app/hf.py`, `app/finetune.py`, `app/heretic.py` | the four features. Each owns its configuration model, its job assembly and its log parser. |
| `app/routers/*.py` | thin HTTP over the above: validate, call, shape the response. No business logic lives here. |
| `app/workers/*.py` | programs that run *inside* containers — the downloader and the fine-tuning trainer. Mounted read-only at `/worker`; they share no imports with the host process. |
| `web/` | vanilla ES modules, one per tab. No framework, no bundler, no CDN. |

## How a job runs

Downloads, image builds, fine-tuning runs and Heretic runs are all the same
shape, which is why the Jobs tab can render a kind it has never heard of.

```
feature module builds a JobSpec  (kind, image, command, env, mounts, meta)
        │
        ▼
jobs.manager.submit  ──>  row in `jobs` (pending)
        │
        ▼
docker run -d  ──>  container name recorded, status running
        │
        ├── docker logs -f  (chunked reads, \r-aware)
        │        │
        │        ├── settled lines appended to <state>/logs/<job_id>.log
        │        │   (a live progress bar is checkpointed every few seconds)
        │        ├── every line published on the job's event topic
        │        └── the kind's registered parser turns some lines into progress
        │
        ▼
container exits  ──>  status succeeded/failed, exit code, result recorded
```

The log is read in fixed-size chunks and split on `\r` as well as `\n`, because
vLLM's and tqdm's progress bars redraw with a bare carriage return; a
`readline()` loop stalls on them and the UI looks frozen mid-load.

A progress parser is registered with `@jobs.register_parser("<kind>")` and
returns only the keys it changed. The manager merges them into the job's
progress dict and publishes the result, so `percent`, `step`, `loss`, `trial` or
anything else a view wants to show is just a key that some parser puts there.

## Streams

Every live surface in the UI is server-sent events, not polling and not
WebSocket: the flow is one-directional, `EventSource` reconnects on its own, and
uvicorn was measured flushing each frame with no buffering. The server side is
`sse_starlette`'s `EventSourceResponse` with a 15-second ping, so an idle stream
survives a silent phase — a torch.compile can produce fifteen seconds of nothing
at all.

Job log streams replay before they follow: the endpoint sends the tail of
`<state>/logs/<job_id>.log`, then a `status` frame, then subscribes to the job's
topic. A browser that connects late, or reconnects, therefore sees the whole run
rather than only what happened after it arrived. A job that has already finished
gets an `end` frame and the client closes the stream itself.

Resume across a *dashboard* restart is server-side: `jobs.manager.reconcile()`
takes the log file's mtime, subtracts five seconds and passes it to
`docker logs --since`, so a few lines are duplicated and none are lost.

One sharp edge to know before adding a stream: `EventSource` treats a non-200
response as a permanent failure and never reconnects. `/api/jobs/{id}/stream`
returns 404 for an unknown id, which is correct for the views that call it
(they stream a job they just listed) but would silently dead-end a pane that
guessed an id.

## Generated schemas

Three files under `app/data/` are checked in but generated, and they are why the
UI does not hard-code any engine flag or sampling parameter.

`vllm_args.json` comes from `tools/gen_vllm_schema.py`, which runs the
configured vLLM image twice: once to introspect the argparse parser for
structure (dest, type, default, choices, group) and once to scrape
`vllm serve --help=all` for prose, because vLLM builds most of its help text
lazily and the parser objects carry empty help strings. The Serve tab renders
its form from the result, so the form always matches the binary that will run.

`llamacpp_args.json` comes from `tools/gen_llamacpp_schema.py` and is the same
document shape, so one renderer serves both forms. It is a single scrape rather
than two passes: llama.cpp has no argparse to introspect — its flag table is
hand-rolled C++ — but its `--help` is printed by one formatter with fixed column
constants, which carries the flags, metavar, prose, default and environment
variable together. It needs no GPU, unlike the vLLM generator. Two things it
cannot scrape and infers by name: which `--no-` spelling negates which flag
(llama.cpp spells its negations rather than deriving them, so a guess is an
argument the binary rejects) and the undocumented `LLAMA_ARG_NO_*` overrides.

`sampling_params.json` comes from `tools/gen_sampling_schema.py`, which reads a
live server's `/openapi.json`. At runtime the Playground re-fetches this per
endpoint and falls back to the checked-in copy; the file matters because vLLM
0.24 replaced `guided_json` and friends with a single `structured_outputs`
object, and the request model accepts unknown keys silently — a UI that emitted
the old names would look like it worked while generating unconstrained text.

## Containers the dashboard did not start

`servers.discover_foreign()` lists every running container that `engines.recognise()`
identifies as an engine and which the dashboard does not manage. They appear on
the Overview and Serve pages beside managed servers, they can be stopped from the
UI, their endpoints are offered to the Playground, and — most importantly — what
they hold counts against the memory budget. The alternative is a memory picture
that is true about this app and false about the machine.

Recognition asks `Config.Cmd` first and only then `Config.Entrypoint + Cmd`. The
fallback exists because an image is free to put the program in its ENTRYPOINT and
leave bare flags in Cmd — the upstream llama.cpp server images do — so a container
whose argv starts with `-m` is otherwise recognisable as nothing at all. Asking
Cmd first means the fallback can only add a container to the picture, never move
one that was already in it, and `ContainerState.command` keeps the meaning every
other parser in the codebase depends on.

Managed containers are named `<LLMD_CONTAINER_PREFIX><kind>-<id>`, where the kind
is the engine's name — so an existing vLLM row keeps `llmd-vllm-<id>` and a
llama.cpp one is `llmd-llamacpp-<id>`. The name is recomputed from the row rather
than stored, which is why changing a saved server's engine while its container
runs is refused: the rename would leave a live process nothing could address.
Managed containers always get
restart policy `no`. On a unified-memory host a crash-looping engine that
re-reserves 60 GiB on every restart is worse than a stopped one, and the
watchdog needs a kill to stay dead.

## Frontend

`web/js/app.js` is a tab router. Each tab is an ES module exporting
`render(container, ctx)` and optionally `dispose()`, imported on first visit, so
one broken view cannot stop the rest of the dashboard from loading. `ctx` gives
a view the API client, the cached system info, the latest telemetry frame, and
functions to set a tab badge or navigate.

Views build DOM with the `h()` helper from `web/js/ui.js` and never interpolate
data into `innerHTML`. Shared components — panels, notices, stat tiles, log
boxes, modals, toasts — live in `ui.js`, and shared styling in
`web/css/app.css`; a view that needs a new class adds it there rather than
inlining styles.

## Adding a feature

The shape is fixed enough to follow mechanically:

1. A module under `app/` with a pydantic config model, a `build_job()` that
   returns a `JobSpec`, and a `@jobs.register_parser` for its log lines.
2. A router under `app/routers/` that validates input and calls it, registered
   in `app/main.py`.
3. A view under `web/js/views/` and an entry in `TABS` in `web/js/app.js`.
4. If it loads a model, a memory pre-flight that returns a `safety.Verdict` —
   a utilisation check for a vLLM engine, an estimate against MemAvailable for
   anything else — surfaced before the launch button, not after.
5. If it needs a new table, extend `SCHEMA` in `app/db.py`; the startup
   `executescript` is idempotent.

## Tests

`tests/` runs against the real modules, in a throwaway state directory, with
docker calls monkeypatched where a test needs one:

```
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

`tests/test_frontend.py` is a static check over the ES modules — it catches a
view importing a helper `ui.js` does not export, which is the failure mode a
build-step-free frontend otherwise only shows you in the browser console.
