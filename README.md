# llm-dashboard

A single web UI for running large language models on one machine — specifically
a DGX Spark, though nothing but the defaults is tied to it. It unifies four
things that are otherwise four sets of shell scripts:

* **Serving.** Define, launch and watch vLLM servers as containers, with a
  parameter form generated from the vLLM image itself.
* **Models.** Search the HuggingFace Hub, see what a repo will cost before you
  pull it, and manage the shared local cache.
* **Fine-tuning.** QLoRA training with Unsloth, from an uploaded JSONL to an
  adapter or a merged model you can serve from the same UI.
* **Decensoring.** Heretic's automated abliteration run headlessly, with its
  Optuna search visible while it happens.

Underneath all four is one thing they share and one thing they fight over: the
host's memory. On this hardware GPU memory *is* host memory, so every launch
goes through a budget check that counts every engine on the box, including ones
the dashboard did not start. Read [docs/MEMORY.md](docs/MEMORY.md) before you
serve anything.

It is a single-user tool with no authentication. Do not expose it to a network
you do not control.

## The seven tabs

**Overview** answers, in order: how much memory is left before the machine
locks up, what is holding it, and what is running. The memory panel is first and
largest on purpose. It names every tenant — managed servers, hand-launched
containers, anything else holding GPU memory — with the utilisation fraction
each has claimed and the largest fraction you could still launch. Below that:
running servers and jobs, GPU telemetry that actually works on GB10 (utilisation,
temperature, power draw, clocks; not memory, which the driver will not report),
disk on the model cache, and any kill the memory watchdog has made.

**Models** searches the Hub and tells you the download size before you commit,
from the repo's own file tree rather than a guess. Pulls run as jobs — the
cache is root-owned, so a download happens inside a container and streams its
byte progress back — and several can be in flight at once. The same page lists
what is already cached, with per-revision sizes, and can delete a repo. Gated
repos are flagged before you queue a pull, and this is where the HuggingFace
token lives if you do not want it in the environment.

**Serve** is the vLLM control surface. Every one of the image's flags is
available: the form is generated from the binary that will run, so it cannot
drift from it, with the memory, scheduling and LoRA parameters promoted to the
top and the remaining two hundred searchable underneath. Three presets carry
their provenance in the UI — two are configurations measured on this host, the
third is derived from them and says so. Before a launch you get the exact
`docker run` command, and a verdict from the memory budget: fits, tight, or
refused with the reason and the largest value that would work. Refusals can be
forced, deliberately, from a separate button. Once running, each server shows
its health, its vLLM metrics (KV cache usage, running and waiting requests,
prefix cache hit rate) and its logs.

**Playground** chats against any healthy endpoint — managed or not — with the
full sampling surface built from that endpoint's own OpenAPI schema, so the
controls are the parameters the server actually accepts. Responses stream, a
parameter set can be saved as a preset, and transcripts are kept so you can come
back to a conversation. A preset saved against an older engine is migrated and
flagged rather than silently sent to a server that no longer has those fields.

**Fine-tune** runs Unsloth QLoRA. Upload a JSONL and it is validated and
shape-detected before it is stored; point at a Hub dataset instead if you
prefer. The form is grouped by what it affects, with the ranks vLLM will refuse
to serve marked as such. A pre-flight estimates the run's peak memory against
what the host has free, because fine-tuning takes no utilisation fraction and
cannot be capped the way an engine can. While it runs you get the loss curve and
step progress; when it finishes, one button turns the result into a server
definition, with the LoRA flags filled in correctly.

**Heretic** does the same for abliteration: pick a model, set the trial budget,
and watch the Optuna search — refusal rate against KL divergence, trial by
trial. The run is fully headless (upstream's interactive menus are configured
away), the study is checkpointed so a run can resume, and the export is either a
merged model or a bare adapter. As with fine-tuning, a finished run can be
handed straight to Serve.

**Jobs** is every long-running container in one place: downloads, image builds,
fine-tunes, Heretic runs. Live logs, progress, cancellation and history. It
survives a dashboard restart — jobs are detached containers, and the dashboard
reattaches to their logs when it comes back.

## Requirements

* Linux with a working `docker` (no sudo required, but your user must be able to
  talk to the daemon) and an NVIDIA runtime.
* Python 3.11 or newer on the host. 3.12 is what this runs on.
* The vLLM container image, `nvcr.io/nvidia/vllm:26.07-py3` by default. It is
  about 22 GB; pull it before you start if you have not already.
* `nvidia-smi` for GPU telemetry. Without it the dashboard still works and the
  GPU tiles simply go quiet.
* Roughly 25 GB of disk for the images, plus whatever the models need.

The dashboard runs as a normal host process, not in a container. It has to:
inside a container, `nvidia-smi` sees only its own PID namespace and could not
tell you how much memory the *other* models are holding.

## Install

```bash
git clone <this repo> ~/llm_dashboard
cd ~/llm_dashboard
cp .env.example .env        # optional; every value has a default
scripts/setup.sh
```

`scripts/setup.sh` creates `.venv`, installs the dependencies with
`--only-binary=:all:` so nothing compiles, creates the state, output and dataset
directories, initialises the database, checks that the checked-in vLLM parameter
schema matches the image you have configured, and then *asks* whether to build
the two worker images. It is safe to run again; every step checks before it
acts. Pass `--with-images` to build without being asked, `--no-images` to skip,
`--dev` to also install pytest and ruff.

## Running it

```bash
scripts/run.sh
```

That is a foreground process; it prints the URL it is listening on (default
`http://0.0.0.0:8700/`). Ctrl-C stops the web process and leaves every container
it started running, which is deliberate — a model that took four minutes to load
should not die because you restarted the UI.

To have it survive a logout, write a systemd *user* unit that runs
`scripts/run.sh` with `Restart=always` — `systemctl --user` needs no sudo. Do
not run it with more than one uvicorn worker: the telemetry poller, the log
followers and the memory watchdog are in-process state.

The memory watchdog can also run on its own, for when you are launching models
by hand and want a net under them:

```bash
scripts/memguard.sh --threshold-mib 12288
```

## Configuration

Everything is an environment variable, read once at import. `.env` in the repo
root is read and exported by all three scripts; the application itself reads
only the process environment, so a value in `.env` affects a dashboard started
through `scripts/run.sh` and not one started some other way.

| Variable | Default | What it does |
|---|---|---|
| `LLMD_HOST` | `0.0.0.0` | Interface the web server binds. There is no authentication. |
| `LLMD_PORT` | `8700` | Web server port. |
| `LLMD_STATE_DIR` | `~/.local/share/llm-dashboard` | Database, job logs, uploads. |
| `LLMD_HF_CACHE` | `/home/user/models/hf-cache` | Shared HuggingFace cache, mounted into every container at `/hf`. |
| `LLMD_OUTPUT_DIR` | `/home/user/models/outputs` | Job artefacts: fine-tunes, Heretic exports. Mounted into server containers at `/outputs`. |
| `LLMD_DATASET_DIR` | `/home/user/models/datasets` | Uploaded training data. |
| `LLMD_VLLM_IMAGE` | `nvcr.io/nvidia/vllm:26.07-py3` | Image used for serving, downloads, and as the base for the two worker images. |
| `LLMD_HERETIC_IMAGE` | `llmd/heretic:latest` | Tag the Heretic tab builds and runs. |
| `LLMD_FINETUNE_IMAGE` | `llmd/finetune:latest` | Tag the fine-tuning tab builds and runs. |
| `LLMD_CONTAINER_PREFIX` | `llmd-` | Prefix for containers the dashboard creates, so it can tell its own from yours. |
| `LLMD_HF_TOKEN` | `$HF_TOKEN`, else empty | Token for gated repos. Can also be stored from the Models tab instead. |
| `LLMD_MEM_RESERVE_GIB` | `32` | Host memory that must remain unclaimed after a launch. Launches that would eat it are refused. |
| `LLMD_MEM_WARN_RESERVE_GIB` | `38` | Softer line: launches past it are allowed with a warning. |
| `LLMD_MEMGUARD_THRESHOLD_MIB` | `10240` | MemAvailable below which the watchdog kills the largest vLLM container. |
| `LLMD_MEMGUARD_ENABLED` | `1` | `0` stops the dashboard from running the watchdog. |
| `LLMD_ROCE_IF` | `enp1s0f0np0` | NCCL/Gloo interface handed to every vLLM container. Only matters for multi-node serving. |
| `LLMD_ROCE_HCA` | `rocep1s0f0` | NCCL IB HCA, likewise. |
| `LLMD_TELEMETRY_INTERVAL` | `2.0` | Seconds between telemetry samples pushed to browsers. |

## How servers are launched

A "server" is a saved parameter set plus a container. Starting one is a single
detached `docker run` that the dashboard shows you in full before it happens:

```
docker run --name llmd-vllm-3 -d --runtime nvidia --gpus all \
  --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/user/models/hf-cache:/hf -v /home/user/models/outputs:/outputs \
  -e HF_HOME=/hf -e NCCL_SOCKET_IFNAME=enp1s0f0np0 ... \
  nvcr.io/nvidia/vllm:26.07-py3 \
  vllm serve unsloth/Qwen3.8-27B-NVFP4 --host 0.0.0.0 --port 8010 \
    --gpu-memory-utilization 0.52 --max-model-len 262144 \
    --kv-cache-dtype fp8 --max-num-seqs 8 --served-model-name qwen3
```

Three things about that are load-bearing.

**Host networking.** vLLM binds the port directly on the host; there is no port
mapping to get wrong, and the port you choose in the form is the port you curl.
The dashboard suggests a free one from 8010 upward, avoiding its own port and
the usual suspects.

**The shared cache.** `/home/user/models/hf-cache` is mounted at `/hf` with
`HF_HOME` pointing at it, so every container — servers, downloads, fine-tunes,
Heretic — reads and writes the same blobs and a model is downloaded once. The
cache tree is root-owned: the host user can read it, which is why cache listing
and "is this already here?" are cheap local operations, but every write happens
inside a container. `/outputs` is mounted too, so a model produced by the
fine-tuning or Heretic tab can be served by path without copying it anywhere.

**Restart policy `no`.** A crash-looping engine that re-reserves 60 GiB every
few seconds is worse than a stopped one, and the watchdog needs a killed
container to stay dead.

The dashboard also adopts vLLM containers it did not start. Anything running a
`vllm serve` command shows up beside managed servers on Overview and Serve: it
can be stopped from the UI, its endpoint is offered to the Playground, and its
`--gpu-memory-utilization` counts against the memory budget. That last point is
the reason the feature exists — a memory picture that is true about this app and
false about the machine is worse than none.

## The memory hazard

The GB10 has no separate framebuffer. `/proc/meminfo` MemTotal and torch's
"total GPU memory" are the same 121.69 GiB, so `--gpu-memory-utilization 0.8` is
a claim on 97 GiB of the host's RAM, and the fractions of co-resident engines
add up with nothing enforcing the sum. Getting it wrong does not slow the
machine down; it freezes it hard enough to need the power button.

The dashboard therefore refuses launches that would leave less than 32 GiB
unclaimed, counts containers it did not start, and runs a watchdog that kills
the largest engine if MemAvailable collapses anyway. Measured on this host:
0.52 for a 27B NVFP4 model beside 0.16 for an 8B embedder is stable; 0.57 for
the LLM alone is stable; 0.80 alone locked the machine.

[docs/MEMORY.md](docs/MEMORY.md) explains the arithmetic, why
`--max-num-seqs` matters more than any rate limit, why a docker memory limit
cannot help you here, and what to do when it goes wrong anyway. It is the one
document to read before using this tool.

## The worker images

Heretic and Unsloth each need a container that the base vLLM image does not
provide. Both are built from `docker/*.Dockerfile` **on top of**
`LLMD_VLLM_IMAGE`, and that is not an implementation detail: the NGC image
already carries working aarch64 CUDA 13 builds of torch, xformers, triton and
flash-attn, which is why nothing in either build compiles. Unsloth's own
DGX Spark Dockerfile builds those from source and takes ten minutes to fail in
interesting ways.

Build them from the Heretic and Fine-tune tabs (a button, streaming the build
log like any other job), from `scripts/setup.sh`, or by hand:

```bash
docker build -t llmd/heretic:latest  -f docker/heretic.Dockerfile  docker/
docker build -t llmd/finetune:latest -f docker/finetune.Dockerfile docker/
```

Each is a single `pip install` layer over the base image. Measured on this box
with `--no-cache` and the base image already present: 28 seconds for Heretic,
14 seconds for fine-tuning, adding 505 MB and 514 MB respectively. The images
report as 22.5 GB because they share the base's 22 GB of layers. If the base
image is *not* pulled, that 22 GB download is the whole cost.

Two build arguments are worth knowing. `HERETIC_REF` (default `master`) selects
the Heretic commit — master is required, because the 1.4.0 release cannot save a
model without a human answering a menu, and the built image records the commit
pip actually resolved so the tab can show it. The fine-tuning Dockerfile takes
optional version pins (`UNSLOTH_VERSION`, `TRL_VERSION`, …); unpinned means
"whatever pip resolves today", and Unsloth ships roughly weekly.

## Where things live

| Path | What |
|---|---|
| `~/.local/share/llm-dashboard/dashboard.db` | SQLite: server definitions, jobs, chat transcripts, presets, datasets, settings. |
| `~/.local/share/llm-dashboard/logs/<job_id>.log` | Full output of every job, kept after it finishes. |
| `~/.local/share/llm-dashboard/uploads/` | Uploaded files staged by the UI. |
| `/home/user/models/hf-cache/` | Shared HuggingFace cache. Root-owned; written only from containers. |
| `/home/user/models/outputs/<name>-<job_id>/` | One directory per job: config, checkpoints, and the exported model or adapter. |
| `/home/user/models/datasets/` | Uploaded JSONL training sets. |
| `app/data/vllm_args.json` | Generated vLLM flag schema. Checked in. |
| `app/data/sampling_params.json` | Generated sampling-parameter schema, used as a fallback. Checked in. |

Deleting the state directory loses your server definitions and job history and
nothing else; the models and outputs are untouched.

## Regenerating the vLLM parameter schema

The Serve form is rendered from `app/data/vllm_args.json`, which is generated
from a vLLM image. When you change `LLMD_VLLM_IMAGE`, regenerate it so the form
matches the binary that will run:

```bash
.venv/bin/python tools/gen_vllm_schema.py \
    --image nvcr.io/nvidia/vllm:26.07-py3 \
    --out app/data/vllm_args.json
```

It runs the image twice — once to introspect argparse for structure, once to
scrape `vllm serve --help=all` for the prose, because vLLM builds its help text
lazily and the parser objects carry empty help strings — and prints how many
flags it found and how many it could describe. Against the image above it takes
about fifteen seconds. It needs the GPU — vLLM will not build its config
dataclasses without a visible device — so it will not work on a machine that has
none. `scripts/setup.sh` notices when the checked-in schema does not match the
configured image and offers to run it for you.

The sampling schema has an equivalent, pointed at a *running* server:

```bash
.venv/bin/python tools/gen_sampling_schema.py --url http://localhost:8000 \
    --out app/data/sampling_params.json
```

The Playground re-fetches this per endpoint at runtime, so the checked-in file
only matters before any server is up.

## Troubleshooting

**A container was OOM-killed.** The Serve tab shows it as `oom-killed` (docker
exit 137, `OOMKilled: true`) and the Jobs tab flags the same thing on a job; the
Overview page shows whether the dashboard's own watchdog did it and why. If the
watchdog did, MemAvailable had fallen below its threshold and it killed the
largest engine to keep the machine responsive. If it did not, the kernel did,
and something else on the box took the memory. Either way the fix is
arithmetic, not retrying: check the budget panel, lower
`--gpu-memory-utilization`, lower `--max-num-seqs`, or stop something. Killed
containers deliberately do not restart themselves.

**A server sits in "loading" forever.** vLLM does not bind its port until the
weights are read and the CUDA graphs are captured, so a refused connection means
"still loading", not "broken" — on a 27B model that is minutes, and the
dashboard shows `loading` rather than `running` for exactly that window. Open
its logs. Three things look like a hang and are not: a multi-gigabyte download
into the cache (the log shows the fetch), a torch.compile phase that produces no
output at all for fifteen seconds or more, and CUDA graph capture. What is
genuinely stuck looks different: a repeated allocation error, or a log that
stopped mid-load while the host's memory kept falling. If the log ends with a
free-memory complaint from vLLM's own pre-flight, the launch was legal by
budget but the host's *free* memory (as opposed to available) was too low
because of page cache — stop something and try again.

**A gated HuggingFace repo.** The Models tab marks gated repos before you queue
a pull, because a gated repo's metadata and file list are readable anonymously
and only the actual download fails, with a 401. Request access on the model's
Hub page, then give the dashboard a token that has it: `LLMD_HF_TOKEN` (or
`HF_TOKEN`) in the environment, or stored from the Models tab, which keeps it in
the database instead. The two are not equivalent. A stored token is used for Hub
search and for download jobs; only a token in the environment is passed into
server, fine-tuning and Heretic containers. So a stored token is enough to
*pull* a gated model into the cache, from where anything can serve it, but a
server pointed straight at a gated repo it has not cached needs the environment
variable. A repo that returns "not accessible with this token" may equally be
private or nonexistent — the Hub does not distinguish, by design.

**A model that needs `trust_remote_code`.** For serving, it is a checkbox in the
Serve form's model section (`--trust-remote-code`); nothing else is needed. For
Heretic it is a hard stop, and the pre-flight says so before you launch:
transformers asks for consent on stdin, and a detached container has no terminal
to answer with, so the job would fail minutes in. The dashboard detects this
from `auto_map` in the model's `config.json`.

## Development

```bash
scripts/setup.sh --dev
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes how the pieces fit: the
process topology and why it is a host process, the module layering, how a job
becomes a container, and what to write where when adding a feature.
