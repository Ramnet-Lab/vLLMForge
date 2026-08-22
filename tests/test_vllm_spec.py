"""The parameter model is generated from the image, so these tests pin the
translation from a stored dict to a command line — the part that is ours."""

from __future__ import annotations

from app import vllm_spec


def test_schema_loaded():
    schema = vllm_spec.schema()
    assert schema["args"], "vllm_args.json is missing or empty; run tools/gen_vllm_schema.py"
    assert schema["vllm_version"].startswith("0.")


def test_build_argv_reproduces_the_hand_written_launch():
    # This is serve-qwen.sh from the hand-written scripts this replaces, expressed as dashboard params.
    params = {
        "served_model_name": "qwen3",
        "max_model_len": 262144,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.52,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "enable_auto_tool_choice": True,
        "tool_call_parser": "qwen3_xml",
    }
    argv = vllm_spec.build_argv("unsloth/Qwen3.8-27B-NVFP4", params, port=8000)
    assert argv[:3] == ["vllm", "serve", "unsloth/Qwen3.8-27B-NVFP4"]
    for expected in (
        "--gpu-memory-utilization", "0.52",
        "--kv-cache-dtype", "fp8",
        "--max-num-seqs", "8",
        "--enable-prefix-caching",
        "--tool-call-parser", "qwen3_xml",
    ):
        assert expected in argv


def test_defaults_are_omitted():
    default = vllm_spec.by_dest()["dtype"]["default"]
    argv = vllm_spec.build_argv("m", {"dtype": default})
    assert "--dtype" not in argv


def test_false_boolean_uses_the_negated_flag_when_one_exists():
    argv = vllm_spec.build_argv("m", {"enable_prefix_caching": False})
    assert "--no-enable-prefix-caching" in argv
    assert "--enable-prefix-caching" not in argv


def test_managed_flags_are_never_emitted():
    argv = vllm_spec.build_argv("m", {"port": 9999, "host": "1.2.3.4"}, port=8000)
    assert argv.count("--port") == 1
    assert "1.2.3.4" not in argv


def test_list_widget_accepts_a_comma_string():
    spec = next(a for a in vllm_spec.schema()["args"] if a["widget"] == "list")
    argv = vllm_spec.build_argv("m", {spec["dest"]: "a, b"})
    assert argv[-2:] == ["a", "b"]


def test_validation_rejects_bad_enum_and_unknown_flags():
    problems = vllm_spec.validate({"kv_cache_dtype": "nope", "not_a_flag": 1})
    assert any("kv-cache-dtype" in p for p in problems)
    assert any("not_a_flag" in p for p in problems)
    assert vllm_spec.validate({"kv_cache_dtype": "fp8"}) == []


def test_ui_model_covers_every_flag_exactly_once():
    ui = vllm_spec.ui_model()
    seen = [
        arg["dest"]
        for section in (*ui["featured"], *ui["advanced"])
        for arg in section["flags"]
    ]
    assert len(seen) == len(set(seen)), "a flag appears in two sections"
    managed = set(ui["managed"])
    expected = {a["dest"] for a in vllm_spec.schema()["args"]} - managed
    assert set(seen) == expected, "the form would hide some flags entirely"


def test_a_size_flag_keeps_its_own_widget():
    """--max-model-len takes 32k and auto as readily as a number. Classifying it
    as an int rendered a number input, which silently blanks anything it cannot
    parse — so the form refused values the backend and vLLM both accept."""
    by_dest = vllm_spec.by_dest()
    assert by_dest["max_model_len"]["widget"] == "size"
    assert by_dest["max_num_batched_tokens"]["widget"] == "size"
    # A plain count stays a plain int.
    assert by_dest["max_num_seqs"]["widget"] == "int"


def test_size_values_survive_validation_and_rendering():
    for value, rendered in (("auto", "auto"), (-1, "-1"), ("32k", "32k"), (131072, "131072")):
        assert vllm_spec.validate({"max_model_len": value}) == []
        argv = vllm_spec.build_argv("org/m", {"max_model_len": value})
        assert argv[argv.index("--max-model-len") + 1] == rendered


def test_a_size_flag_still_rejects_nonsense():
    problems = vllm_spec.validate({"max_model_len": "as long as possible"})
    assert problems and "--max-model-len" in problems[0]


def test_auto_tool_choice_without_a_parser_is_refused():
    """vLLM's own argument validator raises before the model is read, so the
    container is gone in seconds — which from the dashboard looks exactly like
    a launch that failed for a memory reason."""
    problems = vllm_spec.validate({"enable_auto_tool_choice": True})
    assert problems and "needs --tool-call-parser" in problems[0]
    assert vllm_spec.validate(
        {"enable_auto_tool_choice": True, "tool_call_parser": "hermes"}) == []


def test_parser_names_are_checked_against_this_build():
    """The registries are extracted from the image, so a typo is caught at save
    rather than by a KeyError inside the API server after it has started."""
    assert vllm_spec.validate({"tool_call_parser": "hermes",
                               "enable_auto_tool_choice": True}) == []
    bad = vllm_spec.validate({"tool_call_parser": "hermez", "enable_auto_tool_choice": True})
    assert bad and "hermez" in bad[0]
    assert vllm_spec.validate({"reasoning_parser": "qwen3"}) == []
    assert vllm_spec.validate({"reasoning_parser": "qwen4"})


def test_a_parser_without_auto_tool_choice_warns_rather_than_blocks():
    assert vllm_spec.validate({"tool_call_parser": "hermes"}) == []
    warnings = vllm_spec.cross_flag_warnings({"tool_call_parser": "hermes"})
    assert warnings and "ignored unless" in warnings[0]
    assert vllm_spec.cross_flag_warnings(
        {"tool_call_parser": "hermes", "enable_auto_tool_choice": True}) == []
