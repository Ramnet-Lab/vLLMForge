"""The engine seam.

Two engines share one memory budget, one launch lock and one container registry.
What is pinned here is the set of ways that sharing could go wrong — and one of
them, mis-routing a footprint, does not merely misreport: it refuses every
subsequent launch on the machine.
"""

from __future__ import annotations

import pytest

from app import docker_ctl, engines, gguf, llamacpp_spec, safety
from tests.test_gguf import default_keys, write_gguf

GIB = 1024 ** 3
TOTAL = 130_663_006_208


def state(command=None, entrypoint=None, name="c"):
    return docker_ctl.ContainerState(name=name, exists=True, status="running", running=True,
                                     command=command, entrypoint=entrypoint)


# --- recognition ---------------------------------------------------------

@pytest.mark.parametrize("argv,expected", [
    (["vllm", "serve", "org/m"], "vllm"),
    (["vllm", "serve", "org/m", "--gpu-memory-utilization", "0.5"], "vllm"),
    (["llama-server", "-m", "/hf/x.gguf"], "llamacpp"),
    (["/app/llama-server", "--model", "/hf/x.gguf"], "llamacpp"),
    # `./server` is llama.cpp's own historical binary name and is deliberately
    # NOT claimed: `/app/server` is what a Go or Node service is built to, and a
    # recogniser that took those would hand the memory watchdog an operator's
    # own applications to kill. Missing an old binary name costs a budget line.
    (["./server", "-m", "x.gguf"], None),
    (["/app/server", "--config", "/etc/c.yaml"], None),
    # A shell-wrapped launch is one opaque token until somebody who knows the
    # sentinel splits it, and a container that weighs nothing in the budget is
    # how a second launch is admitted on memory already spoken for.
    (["-lc", "cd /x && exec llama-server -m a.gguf -ngl 99"], "llamacpp"),
    (["bash", "-lc", "ray start --head && exec vllm serve m"], "vllm"),
    # Neither.
    (["ray", "start", "--head"], None),
    (["postgres"], None),
    (["python", "-c", "serve"], None),
    # `server` as a bare word is not a program name. Accepting it would make
    # the watchdog a candidate-picker for anything with that argument.
    (["ray", "start", "server"], None),
    ([], None),
    (None, None),
])
def test_which_engine_an_argv_is(argv, expected):
    found = engines.recognise_argv(argv)
    assert (found.name if found else None) == expected


def test_the_entrypoint_is_a_fallback_and_only_a_fallback():
    """The upstream llama.cpp images put the binary in ENTRYPOINT and leave bare
    flags in Cmd, so `command` alone reads as no engine at all. But `command`
    keeps its meaning for every other parser in the codebase, so the fallback
    has to be consulted second — it may add a container to the picture, never
    move one that was already in it."""
    upstream = state(command=["-m", "/models/x.gguf"], entrypoint=["/app/llama-server"])
    assert engines.recognise_argv(upstream.command) is None
    assert engines.recognise(upstream).name == "llamacpp"
    assert upstream.argv == ["/app/llama-server", "-m", "/models/x.gguf"]
    # And a container recognised from Cmd is never re-decided by its entrypoint.
    ours = state(command=["vllm", "serve", "m"], entrypoint=["/usr/local/bin/llmd-entrypoint"])
    assert engines.recognise(ours).name == "vllm"


def test_the_vllm_predicate_was_not_widened():
    """`engines.recognise()` is the general question. Widening is_vllm_command
    instead would have admitted a llama.cpp container to the budget, the
    watchdog and foreign discovery all at once, while `footprint` still priced
    it with vLLM's arithmetic."""
    assert safety.is_vllm_command(["vllm", "serve", "m"])
    assert not safety.is_vllm_command(["llama-server", "-m", "x.gguf"])


def test_an_unknown_engine_name_falls_back_rather_than_raising():
    """A row carrying an engine this build does not have — a downgrade, a
    hand-edited database — must not 500 the Serve page."""
    assert engines.get("nonesuch").name == "vllm"
    assert engines.get(None).name == "vllm"
    assert engines.of({"engine": "llamacpp"}).name == "llamacpp"
    assert not engines.known("nonesuch")


# --- pricing -------------------------------------------------------------

def test_a_llamacpp_config_can_never_be_priced_at_vllms_default():
    """The single mistake that would take the machine's launching down.

    vLLM's pricer charges its default fraction to any argv with no utilisation
    flag, which is every llama.cpp argv there is. On this box that is over
    100 GiB — so one mis-routed server refuses every launch after it.
    """
    llama = engines.get("llamacpp")
    at_default = int(safety.default_util() * TOTAL)
    for params in ({}, {"n_gpu_layers": "99"}, {"ctx_size": "8192"}):
        assert llama.footprint_bytes(params, TOTAL) != at_default
    # And the shared chokepoint routes rather than assumes.
    assert safety.footprint({"n_gpu_layers": "99"}, TOTAL, engine="llamacpp") == 0
    assert safety.footprint({}, TOTAL) == at_default


@pytest.mark.anyio
async def test_a_footprint_is_weights_plus_cache_plus_compute(served_dir):
    path = write_gguf(served_dir / "m-Q4_K_M.gguf", file_bytes=int(4.92 * GIB))
    llama = engines.get("llamacpp")
    header = gguf.read_cached(path)

    resolved = await llama.resolve({"model": str(path), "n_gpu_layers": "99",
                                    "ctx_size": "8192"})
    whole = llama.footprint_bytes(resolved, TOTAL)
    # Weights + a 1 GiB KV cache at 8k + a compute buffer, and nothing wild.
    assert header.file_bytes < whole < header.file_bytes + 3 * GIB

    half = llama.footprint_bytes(
        await llama.resolve({"model": str(path), "n_gpu_layers": "16", "ctx_size": "8192"}),
        TOTAL)
    assert half < whole
    # A quantised cache costs less than an f16 one at the same context.
    quantised = llama.footprint_bytes(
        await llama.resolve({"model": str(path), "n_gpu_layers": "99", "ctx_size": "8192",
                             "cache_type_k": "q8_0", "cache_type_v": "q8_0"}), TOTAL)
    assert quantised < whole


@pytest.mark.anyio
async def test_a_model_that_cannot_be_read_is_unsized_not_free():
    """Zero here is not a claim the launch costs nothing — safety turns an
    unsized llama.cpp launch into a warning that says the guard cannot vouch
    for it, rather than into an approval."""
    llama = engines.get("llamacpp")
    resolved = await llama.resolve({"model": "org/repo:Q4_K_M"})
    assert llama.footprint_bytes(resolved, TOTAL) == 0
    assert "downloads itself" in resolved["_sizing"]
    assert llama.notes(resolved, implicit=False)


def test_llamacpp_declares_no_fraction_and_says_so():
    """Reporting bytes/total as a util would make two engines' Util columns look
    summable. They are not, and an operator will add them."""
    llama = engines.get("llamacpp")
    assert llama.declared_util({"n_gpu_layers": "99"}) is None
    assert llama.implicit_util() is None
    assert engines.get("vllm").implicit_util() == safety.default_util()


# --- argv ----------------------------------------------------------------

def test_the_model_is_a_flag_not_a_positional():
    llama = engines.get("llamacpp")
    argv = llama.build_argv("/hf/hub/models--x/snapshots/abc/m.gguf", {}, port=8010)
    assert argv[0] == "llama-server"
    assert argv[argv.index("-m") + 1] == "/hf/hub/models--x/snapshots/abc/m.gguf"
    assert argv[argv.index("--port") + 1] == "8010"
    # Always on: /metrics 404s without it and the Metrics tab could never fill.
    assert "--metrics" in argv


def test_a_hub_reference_goes_to_hf_and_a_path_goes_to_m():
    llama = engines.get("llamacpp")
    assert "-hf" in llama.build_argv("org/repo:Q4_K_M", {})
    assert "-m" in llama.build_argv("/outputs/run/gguf/model.gguf", {})
    assert "-m" in llama.build_argv("model.gguf", {})


def test_a_false_boolean_needs_the_spelling_llama_cpp_actually_has():
    """argparse derives --no-<flag>; llama.cpp does not, and emitting a guessed
    negation would be an argument the binary rejects."""
    argv = llamacpp_spec.build_argv("/x.gguf", {"jinja": False})
    assert "--no-jinja" in argv
    # A flag with no negative spelling emits nothing rather than something wrong.
    assert llamacpp_spec.validate({"metrics": False})       # managed anyway
    argv = llamacpp_spec.build_argv("/x.gguf", {"props": False})
    assert "--props" not in argv and "--no-props" not in argv


def test_defaults_are_omitted_and_managed_flags_are_refused():
    argv = llamacpp_spec.build_argv("/x.gguf", {"ctx_size": 0, "n_gpu_layers": 40}, port=8010)
    assert "--ctx-size" not in argv               # 0 is the schema's default
    assert argv[argv.index("--n-gpu-layers") + 1] == "40"
    assert argv.count("--port") == 1
    assert "port is managed by the dashboard" in " ".join(llamacpp_spec.validate({"port": 9}))


def test_ngl_accepts_the_words_it_actually_takes():
    """'auto' and 'all' are its documented values, and a validator that only
    knew integers would reject its own default."""
    assert llamacpp_spec.validate({"n_gpu_layers": "all"}) == []
    assert llamacpp_spec.validate({"n_gpu_layers": "auto"}) == []
    assert llamacpp_spec.validate({"n_gpu_layers": "40"}) == []
    assert llamacpp_spec.validate({"n_gpu_layers": "most"})
    # And 'auto'/'all' mean "not pinned", which is priced as the whole model.
    assert llamacpp_spec.n_gpu_layers({"n_gpu_layers": "all"}) is None
    assert llamacpp_spec.n_gpu_layers({"n_gpu_layers": "40"}) == 40


def test_a_quantised_v_cache_without_flash_attention_is_refused_early():
    """llama.cpp raises during context creation rather than falling back, so the
    container exits minutes in, after the weights have been read."""
    problems = llamacpp_spec.validate({"cache_type_v": "q8_0", "flash_attn": "off"})
    assert any("flash attention" in p for p in problems)
    assert llamacpp_spec.validate({"cache_type_v": "q8_0", "flash_attn": "on"}) == []


def test_every_widget_kind_the_form_renders_is_present():
    """The renderer is shared with the vLLM form, and a schema missing a widget
    kind is a control nobody discovers is broken."""
    widgets = {arg["widget"] for arg in llamacpp_spec.schema()["args"]}
    assert {"bool", "enum", "int", "float", "str", "list", "size"} <= widgets


def test_the_schema_document_has_the_shape_the_form_expects():
    index = llamacpp_spec.by_dest()
    assert index, "the checked-in schema is empty"
    # Every entry carries every key, explicit nulls included — a missing key is
    # an undefined in the renderer rather than a visible default.
    expected = {"dest", "flags", "type", "nargs", "default", "choices", "action", "group",
                "required", "widget", "flag", "negatable", "negative_flag", "accepts", "env",
                "help"}
    for arg in llamacpp_spec.schema()["args"]:
        assert expected <= set(arg), f"{arg['dest']} is missing {expected - set(arg)}"
    model = llamacpp_spec.ui_model()
    assert model["engine"] == "llamacpp" and model["featured"] and model["advanced"]


def test_the_form_covers_every_flag_exactly_once():
    model = llamacpp_spec.ui_model()
    seen = [arg["dest"] for group in ("featured", "advanced")
            for section in model[group] for arg in section["flags"]]
    assert len(seen) == len(set(seen)), "a flag appears in two sections"
    assert set(seen) == set(llamacpp_spec.by_dest()) - set(model["managed"])


def test_a_loading_server_is_the_opposite_shape_in_each_engine():
    """vLLM binds no port until the weights are in; llama-server binds first and
    answers /health with 503. One rule would leave one of them looking broken
    for the several minutes a large model takes."""
    vllm, llama = engines.get("vllm"), engines.get("llamacpp")
    assert vllm.is_loading(reachable=False, healthy=False, models=[])
    assert not vllm.is_loading(reachable=True, healthy=False, models=[])
    assert llama.is_loading(reachable=True, healthy=False, models=[])
    assert not llama.is_loading(reachable=False, healthy=False, models=[])


def test_a_foreign_containers_model_is_read_the_way_its_engine_spells_it():
    vllm, llama = engines.get("vllm"), engines.get("llamacpp")
    assert vllm.model_from_argv(["vllm", "serve", "org/m", "--port", "8000"]) == "org/m"
    assert llama.model_from_argv(["llama-server", "-m", "/hf/x.gguf"]) == "/hf/x.gguf"
    assert llama.model_from_argv(["llama-server", "--model=/hf/y.gguf"]) == "/hf/y.gguf"
    assert llama.model_from_argv(["llama-server", "-hf", "org/repo:Q4_K_M"]) == "org/repo:Q4_K_M"
    # A vLLM argv read as llama.cpp finds nothing rather than something wrong.
    assert llama.model_from_argv(["vllm", "serve", "org/m"]) == ""


def test_pooling_belongs_to_the_engine_that_has_it():
    assert engines.get("vllm").supports_pooling
    assert not engines.get("llamacpp").supports_pooling


# --- what the pricer must not under-charge -------------------------------

@pytest.mark.anyio
async def test_a_cpu_resident_cache_is_not_free(served_dir):
    """The KV cache is deliberately NOT discounted by the offload fraction.

    Only the offloaded layers keep their cache in a framebuffer — but on the
    machine this dashboard is most careful about, GPU memory IS host memory, so
    the cache of a CPU-resident layer lands in the very pool being guarded.
    Discounting it priced `-ngl 0` on a large model at the compute floor alone
    and answered "Fits" for a launch that takes the machine.
    """
    path = write_gguf(served_dir / "big-Q8_0.gguf", keys=default_keys(layers=80),
                      file_bytes=int(40 * GIB))
    llama = engines.get("llamacpp")
    priced = llama.footprint_bytes(
        await llama.resolve({"model": str(path), "n_gpu_layers": "0", "ctx_size": "131072"}),
        TOTAL)
    # Weights come off, the cache does not.
    assert priced > 20 * GIB


@pytest.mark.anyio
async def test_a_draft_model_and_a_projector_are_counted(served_dir):
    """Both are separate GGUFs loaded fully into the same pool, and neither
    appears anywhere in the main model's header."""
    main = write_gguf(served_dir / "main-Q4_K_M.gguf", file_bytes=int(4 * GIB))
    draft = write_gguf(served_dir / "draft-Q4_K_M.gguf", file_bytes=int(2 * GIB))
    llama = engines.get("llamacpp")

    base = {"model": str(main), "n_gpu_layers": "99", "ctx_size": "4096"}
    alone = llama.footprint_bytes(await llama.resolve(dict(base)), TOTAL)
    with_draft = llama.footprint_bytes(
        await llama.resolve(dict(base, spec_draft_model=str(draft))), TOTAL)
    assert with_draft - alone == pytest.approx(2 * GIB, rel=0.01)


@pytest.mark.anyio
async def test_a_model_reference_cannot_reach_outside_the_mounts():
    """The reference comes from a form field and is opened by the dashboard
    process. `..` is not a legal part of one, so folding it is not a
    restriction anyone runs into — and a prefix test alone would hand over
    whatever it resolved to."""
    from app.engines.llamacpp import host_path

    assert host_path("/hf/../../../etc/passwd") is None
    assert host_path("/hf/hub/../../../root/.ssh/id_rsa") is None
    # An absolute path outside the two mounts is not opened on a container's
    # say-so either: a foreign engine's own mounts are unknown to this process.
    assert host_path("/root/.ssh/id_rsa") is None
    assert host_path("/etc/shadow") is None
    # What is legitimate still resolves.
    assert str(host_path("/hf/hub/models--x/snapshots/a/m.gguf")).endswith(
        "hub/models--x/snapshots/a/m.gguf")


def test_a_stashed_argument_set_is_validated_before_it_is_promoted(tmp_path):
    """`args` is validated by the API; `args_by_engine` is not. Promoting the
    stash on an engine switch would otherwise be a way past the check."""
    from app import db, servers

    row = servers.create_server({
        "name": f"stashed-{db.now():.6f}", "model": "org/m", "port": 8461,
        "args": {"gpu_memory_utilization": 0.4},
        "args_by_engine": {"llamacpp": {"not_a_flag": 1, "ctx_size": "1e400"}},
    })
    try:
        servers.update_server(int(row["id"]), {"engine": "llamacpp", "model": "/hf/m.gguf"})
        after = servers.get_server(int(row["id"]))
        assert after["engine"] == "llamacpp"
        assert after["args"] == {}, "nonsense was promoted into the authoritative column"
        # The vLLM set it switched away from is kept, even though the caller
        # never sent a stash on the patch.
        assert after["args_by_engine"]["vllm"] == {"gpu_memory_utilization": 0.4}
    finally:
        servers.delete_server(int(row["id"]))
