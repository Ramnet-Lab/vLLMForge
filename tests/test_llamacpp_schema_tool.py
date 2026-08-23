"""The llama.cpp schema scraper.

It cannot be run here — it needs docker and a built image — but everything it
does to the text is pure, and the accuracy of the checked-in schema depends on
it. The sample below is a byte-accurate reproduction of `common_arg::to_string()`
output: aliases at column 0, the first padded to width 7, help at column 40.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "gen_llamacpp_schema", Path(__file__).resolve().parent.parent / "tools"
    / "gen_llamacpp_schema.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


HELP = (
    "----- common params -----\n"
    "\n"
    "-t,    --threads N                      number of CPU threads to use during generation\n"
    "                                        (default: -1)\n"
    "                                        (env: LLAMA_ARG_THREADS)\n"
    "-ngl,  --gpu-layers, --n-gpu-layers N   max layers in VRAM (default: auto)\n"
    "                                        (env: LLAMA_ARG_N_GPU_LAYERS)\n"
    "-ctk,  --cache-type-k TYPE              KV cache data type for K\n"
    "                                        allowed values: f32, f16, bf16, q8_0, q4_0\n"
    "                                        (default: f16)\n"
    "                                        (env: LLAMA_ARG_CACHE_TYPE_K)\n"
    "--top-k N                               top-k sampling (default: 40, 0 = disabled)\n"
    "--temp, --temperature N                 temperature (default: 0.80)\n"
    "--mmap, --no-mmap                       memory-map the model (default: enabled)\n"
    "-fa,   --flash-attn [on|off|auto]       set Flash Attention use (default: auto)\n"
    "\n"
    "----- sampling params -----\n"
    "\n"
    "--min-p N                               min-p sampling (default: 0.05, 0.0 = disabled)\n"
)


@pytest.fixture(scope="module")
def parsed():
    return {gen.dest_of(e["flags"]): e for e in gen.parse_help(HELP)}


def _widget(entry):
    default = gen.DEFAULT_IN_HELP.search(entry["help"])
    default = default.group(1).strip() if default else None
    allowed = gen.ALLOWED_IN_HELP.search(entry["help"])
    choices = ([c.strip() for c in allowed.group(1).split(",") if c.strip()]
               if allowed else None)
    match = gen.CHOICES_IN_METAVAR.match(entry["metavar"])
    if choices is None and match and "|" in match.group(1):
        choices = [c.strip() for c in match.group(1).split("|")]
    return gen.classify(entry["metavar"], default, choices), default, choices


def test_the_sections_are_read(parsed):
    assert parsed["threads"]["section"] == "common params"
    assert parsed["min_p"]["section"] == "sampling params"


def test_a_float_is_not_turned_into_an_integer(parsed):
    """llama.cpp prints `N` for `--temp` exactly as it does for `--main-gpu`, so
    a metavar-driven classifier makes every sampling default an integer field —
    a form that silently truncates 0.80 to 0."""
    assert _widget(parsed["temp"])[0] == "float"
    assert _widget(parsed["min_p"])[0] == "float"
    assert _widget(parsed["threads"])[0] == "int"
    assert _widget(parsed["top_k"])[0] == "int"


def test_a_default_is_the_value_and_not_the_note_beside_it(parsed):
    """`(default: 40, 0 = disabled)` is one parenthetical carrying two things."""
    assert _widget(parsed["top_k"])[1] == "40"
    assert _widget(parsed["min_p"])[1] == "0.05"
    assert _widget(parsed["threads"])[1] == "-1"


def test_allowed_values_printed_as_prose_still_become_an_enum(parsed):
    """-ctk's metavar is TYPE; its allowed set is a separate help line. Missing
    it renders a free-text box, and a typo is discovered when the container
    exits."""
    widget, default, choices = _widget(parsed["cache_type_k"])
    assert widget == "enum"
    assert choices == ["f32", "f16", "bf16", "q8_0", "q4_0"]
    assert default == "f16"


def test_a_bracketed_metavar_is_an_enum_too(parsed):
    widget, default, choices = _widget(parsed["flash_attn"])
    assert widget == "enum" and choices == ["on", "off", "auto"] and default == "auto"


def test_a_word_default_gets_the_widget_that_can_hold_one(parsed):
    """-ngl's default is 'auto', which a number input cannot hold."""
    assert _widget(parsed["gpu_layers"])[0] == "size"


def test_a_flag_with_no_metavar_is_a_boolean(parsed):
    """The help starts at a fixed column rather than after two spaces, so a
    column-blind split read a short flag's whole help text as its metavar."""
    entry = parsed["mmap"]
    assert entry["metavar"] == ""
    assert _widget(entry)[0] == "bool"
    assert entry["help"].startswith("memory-map the model")
    assert "--no-mmap" in entry["flags"]


def test_the_env_var_is_read_and_kept_out_of_the_prose(parsed):
    entry = parsed["threads"]
    assert gen.ENV_IN_HELP.search(entry["help"]).group(1) == "LLAMA_ARG_THREADS"
    assert "wrapped" not in entry["help"]
    # A wrapped help line is rejoined rather than left as two.
    assert "generation (default: -1)" in entry["help"]
