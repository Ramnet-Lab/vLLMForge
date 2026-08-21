"""Pull a Hub repo into the shared cache, reporting progress as JSON lines.

Runs inside the vLLM image as root, with the cache bind-mounted at /hf and
HF_HOME pointing at it, because that is the only way anything gets written to a
root-owned cache. Invoked as:

    python -u /worker/hf_download.py --repo-id Qwen/Qwen3-0.6B --revision main

stdout carries nothing but marker-prefixed JSON, one object per line; the
library's own chatter and tqdm's redraws go to stderr, so the dashboard's line
parser never has to guess what it is looking at.

Two details that are easy to get wrong and silent when you do:

* the progress class must subclass tqdm.auto.tqdm, *not* huggingface_hub's own
  tqdm. huggingface_hub injects `disable=None` into its own subclasses, which
  makes tqdm auto-detect a TTY and emit nothing at all in a piped container;
* huggingface_hub passes a `name=` kwarg that vanilla tqdm rejects, so it has
  to be dropped in __init__.

Only the standard library and huggingface_hub are available here — this file is
mounted into an image the dashboard does not build.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import traceback

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm as vanilla_tqdm

PROGRESS_MARKER = "@@PROGRESS@@"
RESULT_MARKER = "@@RESULT@@"
EMIT_INTERVAL = 0.5


def emit(marker: str, payload: dict) -> None:
    sys.stdout.write(f"{marker} {json.dumps(payload)}\n")
    sys.stdout.flush()


class Reporter:
    """Collapses snapshot_download's three aggregate bars into one progress line."""

    def __init__(self) -> None:
        self.phase = "starting"
        self.plan_bytes = 0
        self.files_total = 0
        self.files_done = 0
        self.net_bytes = 0
        self.disk_bytes = 0
        self.bar_bytes = 0
        self.speed_bps = 0.0
        self.started = time.monotonic()
        self._last_emit = 0.0
        self._speed_at = self.started
        self._speed_bytes = 0

    def sample(self, bar: vanilla_tqdm) -> None:
        desc = bar.desc or ""
        if bar.unit == "B":
            # Two byte bars: network ("Downloading bytes") and disk
            # ("Reconstructing ..."). They legitimately disagree — the network
            # runs ahead of the writer.
            if desc.startswith("Downloading"):
                self.net_bytes = bar.n
            else:
                self.disk_bytes = bar.n
            self.bar_bytes = max(self.bar_bytes, int(bar.total or 0))
        else:
            self.files_done = bar.n
            self.files_total = max(self.files_total, int(bar.total or 0))

    @property
    def downloaded(self) -> int:
        # Xet can satisfy part of a file from chunks it already holds, so the
        # network bar legitimately finishes short of the repo total; the writer
        # bar is the one that always lands exactly on it.
        if self.phase == "done":
            return max(self.net_bytes, self.disk_bytes)
        return self.net_bytes or self.disk_bytes

    @property
    def total(self) -> int:
        # The bars start at total=0 and grow as each file registers, so the
        # dry-run figure is the only stable denominator early on.
        return self.plan_bytes or self.bar_bytes

    def _tick_speed(self, now: float) -> None:
        elapsed = now - self._speed_at
        if elapsed < 0.4:
            return
        rate = max(0, self.downloaded - self._speed_bytes) / elapsed
        self.speed_bps = rate if self.speed_bps == 0.0 else 0.6 * self.speed_bps + 0.4 * rate
        self._speed_at, self._speed_bytes = now, self.downloaded

    def publish(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < EMIT_INTERVAL:
            return
        self._last_emit = now
        self._tick_speed(now)
        total, done = self.total, self.downloaded
        if self.phase == "done":
            percent = 100.0
        elif total:
            percent = min(100.0, 100.0 * done / total)
        elif self.files_total:
            percent = 100.0 * self.files_done / self.files_total
        else:
            percent = 0.0
        remaining = max(total - done, 0)
        emit(
            PROGRESS_MARKER,
            {
                "phase": self.phase,
                "downloaded_bytes": done,
                "written_bytes": self.disk_bytes,
                "total_bytes": total,
                "percent": round(percent, 2),
                "files_done": self.files_done,
                "files_total": self.files_total,
                "speed_bps": round(self.speed_bps),
                "elapsed": round(now - self.started, 1),
                "eta": round(remaining / self.speed_bps) if self.speed_bps > 1 else None,
            },
        )


reporter = Reporter()


class JsonTqdm(vanilla_tqdm):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        reporter.sample(self)

    def update(self, n: float = 1) -> bool | None:
        result = super().update(n)
        reporter.sample(self)
        reporter.publish()
        return result

    def close(self) -> None:
        reporter.sample(self)
        reporter.publish(force=True)
        super().close()


def plan(repo_id: str, revision: str, patterns: dict) -> dict | None:
    """Price the pull before starting it, so percentages mean something."""
    if "dry_run" not in inspect.signature(snapshot_download).parameters:
        return None
    files = snapshot_download(repo_id, revision=revision, dry_run=True, **patterns)
    return {
        "files_total": len(files),
        "total_bytes": sum(f.file_size for f in files if f.will_download),
        "cached_bytes": sum(f.file_size for f in files if f.is_cached),
        "snapshot_bytes": sum(f.file_size for f in files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a HuggingFace repo into the cache")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--allow", action="append", default=[], metavar="GLOB")
    parser.add_argument("--ignore", action="append", default=[], metavar="GLOB")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="price the pull and exit")
    args = parser.parse_args()

    patterns = {
        "allow_patterns": args.allow or None,
        "ignore_patterns": args.ignore or None,
    }

    try:
        reporter.phase = "planning"
        reporter.publish(force=True)
        estimate = plan(args.repo_id, args.revision, patterns)
        if estimate:
            reporter.plan_bytes = estimate["total_bytes"]
            reporter.files_total = estimate["files_total"]
        if args.dry_run:
            emit(RESULT_MARKER, {"phase": "planned", "repo_id": args.repo_id, **(estimate or {})})
            return 0

        reporter.phase = "downloading"
        reporter.publish(force=True)
        path = snapshot_download(
            args.repo_id,
            revision=args.revision,
            tqdm_class=JsonTqdm,
            max_workers=args.max_workers,
            **patterns,
        )
    except Exception as exc:
        traceback.print_exc()
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    reporter.phase = "done"
    reporter.publish(force=True)
    emit(
        RESULT_MARKER,
        {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "path": path,
            "downloaded_bytes": reporter.downloaded,
            "total_bytes": reporter.total,
            "files_total": reporter.files_total,
            "elapsed": round(time.monotonic() - reporter.started, 1),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
