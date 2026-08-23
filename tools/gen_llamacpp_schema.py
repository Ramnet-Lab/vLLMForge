#!/usr/bin/env python3
"""Generate a machine-readable schema of `llama-server` flags from an image.

The sibling of tools/gen_vllm_schema.py, and it writes the identical document
shape so app/llamacpp_spec.py and the frontend's form renderer need no branch:

    python3 tools/gen_llamacpp_schema.py --image llmd/llamacpp:latest \
        --out app/data/llamacpp_args.json

It differs from the vLLM generator in two ways that matter.

**One pass, not two.** vLLM builds its help text lazily out of dataclass
docstrings, so its schema has to be introspected for structure and scraped for
prose. llama.cpp has no argparse to introspect — `common/arg.cpp` is a
hand-rolled C++ table — but its `--help` is emitted by one formatter with fixed
constants, so a single scrape gets everything: flags, metavar, help, default and
env var. The format, from `common_arg::to_string()`:

    ----- common params -----

    -t,    --threads N                      number of CPU threads (default: -1)
                                            (env: LLAMA_ARG_THREADS)

  * sections are `----- <name> params -----`, four of them for llama-server
  * a flag line starts at column 0; help starts at column 40 and wraps at 70
  * the first alias is padded to width 7; later ones follow, comma separated
  * a metavar follows the last alias after a space
  * the default is embedded in the prose as `(default: X)`, never a column
  * the env var, when there is one, is a trailing `(env: NAME)` line

**No GPU needed.** vLLM refuses to build its config dataclasses without a
visible device, so its generator has to hand the GPU through. `llama-server
--help` prints a table and exits, so this runs anywhere docker does.

Two things `--help` does not print, and this therefore cannot scrape:

  * the negative spelling of a boolean. llama.cpp spells its negations rather
    than deriving them, and `--no-mmap` appears as its own entry — so the
    pairing is inferred here by name, and the result is worth a glance.
  * `LLAMA_ARG_NO_*`, which the parser accepts for any negatable flag but never
    documents. Nothing here needs it; it is noted so its absence is not read as
    an oversight.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# common_arg::to_string(): n_leading_spaces = 40, n_char_per_line_help = 70.
HELP_COLUMN = 40

SECTION = re.compile(r"^-{5}\s+(.+?)\s+-{5}$")
# A flag line: one or more comma-separated aliases at column 0, then an optional
# metavar. Anything indented to the help column is continuation.
FLAG_LINE = re.compile(r"^(-{1,2}[^\s,]+(?:,\s*-{1,2}[^\s,]+)*)(?:\s+(.*))?$")
# `(default: X)` — but llama.cpp routinely writes `(default: 40, 0 = disabled)`
# and `(default: -1, -1 = infinity)`, one parenthetical carrying the value and a
# note about it. Capturing the whole thing made `int("40, 0 = disabled")` fail
# and left the prose as the default, which the form then renders as a value and
# `_is_default` compares every submission against.
DEFAULT_IN_HELP = re.compile(r"\(default:\s*([^,)]+?)\s*(?:,[^)]*)?\)")
ENV_IN_HELP = re.compile(r"\(env:\s*([A-Z0-9_]+)\)")
ALLOWED_IN_HELP = re.compile(r"allowed values?:\s*([^.(]+)", re.I)
CHOICES_IN_METAVAR = re.compile(r"^[\[{](.+?)[\]}]$")


def run_help(image: str) -> str:
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "llama-server", image, "--help"],
        capture_output=True, text=True,
    )
    text = proc.stdout + proc.stderr
    if "-----" not in text:
        sys.exit(f"llama-server --help produced nothing recognisable:\n{text[-4000:]}")
    return text


def parse_help(text: str) -> list[dict]:
    """Every flag entry, as (aliases, metavar, help, section)."""
    entries: list[dict] = []
    section = "common params"
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            current["help"] = re.sub(r"\s+", " ", " ".join(current["help_parts"])).strip()
            del current["help_parts"]
            entries.append(current)
        current = None

    for raw in text.splitlines():
        line = raw.rstrip()
        found = SECTION.match(line.strip())
        if found:
            flush()
            section = found.group(1).strip()
            continue
        if not line.strip():
            continue
        if line.startswith(" " * HELP_COLUMN) or (current and line.startswith("  ")):
            if current is not None:
                current["help_parts"].append(line.strip())
            continue

        match = FLAG_LINE.match(line)
        if not match:
            continue
        flush()
        aliases = [part.strip() for part in match.group(1).split(",") if part.strip()]
        # Split by COLUMN, not by a run of spaces. The formatter pads to a fixed
        # column rather than separating with two spaces the way argparse does,
        # so partitioning on "  " read the whole help text of every short flag
        # line as that flag's metavar — which then classified it as a string.
        invocation_end = match.end(1)
        metavar = line[invocation_end:HELP_COLUMN].strip() if invocation_end < HELP_COLUMN else ""
        rest = line[HELP_COLUMN:].strip() if len(line) > HELP_COLUMN else ""
        if invocation_end >= HELP_COLUMN:
            # The invocation ran past the column, so the formatter put the help on
            # the next line and everything after the aliases here is the metavar.
            metavar = line[invocation_end:].strip()
        current = {"flags": aliases, "metavar": metavar, "section": section,
                   "help_parts": [rest] if rest else []}
    flush()
    return entries


def dest_of(flags: list[str]) -> str:
    longs = [f for f in flags if f.startswith("--") and not f.startswith("--no-")]
    name = (longs[0] if longs else flags[0]).lstrip("-")
    return name.replace("-", "_")


def classify(metavar: str, default: str | None, choices: list[str] | None) -> str:
    """Which control the form should render. Same seven-value vocabulary the
    vLLM schema uses, so one renderer serves both.

    The DEFAULT's shape decides between int and float, not the metavar. That is
    the whole trick, and it is not a nicety: llama.cpp prints `N` for
    `--temp N (default: 0.80)` exactly as it does for `--main-gpu N (default:
    0)`, so a metavar-driven classifier turns temperature, top-p, min-p,
    repeat-penalty and a dozen more into integer fields — a form that silently
    truncates every sampling default an operator types.
    """
    if choices:
        return "enum"
    if not metavar:
        return "bool"
    upper = metavar.upper()
    if upper in ("N", "COUNT", "INDEX", "PORT", "P", "F", "SCALE", "T"):
        number = _number(default)
        if number is None:
            # A non-numeric default is a word: `-ngl auto`, `-fa auto`. `size` is
            # the widget that can hold a word as well as a number, which a plain
            # number input cannot.
            return "size" if default else "int"
        return "float" if isinstance(number, float) else "int"
    if "," in metavar or upper.endswith("..."):
        return "list"
    return "str"


def _number(text: str | None) -> int | float | None:
    """The default as a number, keeping int and float apart. None if it is neither."""
    if text is None:
        return None
    raw = str(text).strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        value = float(raw)
    except ValueError:
        return None
    return None if value != value or value in (float("inf"), float("-inf")) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="llmd/llamacpp:latest")
    parser.add_argument("--out", default="app/data/llamacpp_args.json")
    args = parser.parse_args()

    if not shutil.which("docker"):
        sys.exit("docker not found on PATH")

    print(f"reading {args.image} llama-server --help ...", file=sys.stderr)
    text = run_help(args.image)
    entries = parse_help(text)
    print(f"  {len(entries)} flag entries", file=sys.stderr)

    version = ""
    for line in text.splitlines():
        if "version:" in line.lower() or line.strip().startswith("build"):
            version = line.strip()
            break

    by_dest: dict[str, dict] = {}
    negations: dict[str, str] = {}
    sections: dict[str, str] = {}

    for entry in entries:
        flags, help_text = entry["flags"], entry["help"]
        sections.setdefault(entry["section"], "")
        negative = next((f for f in flags if f.startswith("--no-")), None)
        positives = [f for f in flags if not f.startswith("--no-")]
        if negative and not positives:
            # A standalone negation, e.g. `--no-context-shift` on its own line.
            negations[negative[len("--no-"):].replace("-", "_")] = negative
            continue

        dest = dest_of(positives or flags)
        default_match = DEFAULT_IN_HELP.search(help_text)
        default = default_match.group(1).strip() if default_match else None
        env_match = ENV_IN_HELP.search(help_text)
        metavar = entry["metavar"]
        choice_match = CHOICES_IN_METAVAR.match(metavar)
        choices = ([c.strip() for c in choice_match.group(1).split("|")]
                   if choice_match and "|" in choice_match.group(1) else None)
        if choices is None and choice_match and "," in choice_match.group(1):
            choices = [c.strip() for c in choice_match.group(1).split(",")]
        if choices is None:
            # Several flags print their allowed set as prose rather than as a
            # metavar — `-ctk TYPE` with "allowed values: f32, f16, …" on its own
            # line, and `--load-mode MODE` likewise. Without this they render as
            # free-text boxes, and a typo is discovered when the container exits.
            allowed = ALLOWED_IN_HELP.search(help_text)
            if allowed:
                choices = [c.strip() for c in allowed.group(1).split(",") if c.strip()]

        widget = classify(metavar, default, choices)
        prose = ENV_IN_HELP.sub("", help_text).strip()
        by_dest[dest] = {
            "dest": dest,
            "flags": flags,
            "type": {"int": "int", "size": "int", "float": "float"}.get(widget),
            "nargs": "+" if widget == "list" else None,
            "default": _coerce(default, widget),
            "choices": choices,
            "action": "_StoreTrueAction" if widget == "bool" else "_StoreAction",
            "group": entry["section"],
            "required": False,
            "widget": widget,
            "flag": next((f for f in flags if f.startswith("--")), flags[0]),
            "negatable": bool(negative),
            "negative_flag": negative,
            "accepts": ["auto", "all"] if widget == "size" else [],
            "env": env_match.group(1) if env_match else None,
            "help": prose,
        }
        if negative:
            negations[dest] = negative

    for dest, negative in negations.items():
        if dest in by_dest and not by_dest[dest]["negative_flag"]:
            by_dest[dest]["negative_flag"] = negative
            by_dest[dest]["negatable"] = True

    payload = {
        "image": args.image,
        "llamacpp_version": version or "unknown",
        "source": f"Scraped from `llama-server --help` of {args.image}.",
        "sections": sections,
        "args": sorted(by_dest.values(), key=lambda a: (a["group"], a["dest"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)", file=sys.stderr)
    print("Check the negatable flags by hand: llama.cpp spells its negations and "
          "this pairs them by name.", file=sys.stderr)


def _coerce(default: str | None, widget: str):
    if default is None:
        return None
    text = default.strip().strip("`\"'")
    if widget == "bool":
        return text.lower() in ("true", "enabled", "on", "1")
    if widget in ("int", "float"):
        try:
            return int(text) if widget == "int" else float(text)
        except ValueError:
            return text
    return text


if __name__ == "__main__":
    main()
