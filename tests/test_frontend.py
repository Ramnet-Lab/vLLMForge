"""Static checks on the frontend.

There is no build step and no bundler, so nothing stands between a typo and a
blank tab. These are the two failures that actually happen when several people
write views against one toolkit: unbalanced delimiters, and importing a helper
that does not exist. A browser is the only real syntax checker, but neither of
those needs one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
JS_FILES = sorted(WEB.rglob("*.js"))
VIEWS = sorted((WEB / "js" / "views").glob("*.js"))

PAIRS = {")": "(", "]": "[", "}": "{"}


def strip_js(source: str) -> str:
    """Blank out comments, strings, template literals and regex literals.

    Everything that survives is code, so delimiter counting means something.
    Template literals keep their ${...} substitutions, which are code too.
    Newlines inside what gets blanked are preserved, so a reported line number
    still points at the real line.
    """
    out: list[str] = []
    index, length = 0, len(source)
    while index < length:
        char = source[index]
        two = source[index:index + 2]

        if two == "//":
            end = source.find("\n", index)
            index = length if end == -1 else end
            continue
        if two == "/*":
            end = source.find("*/", index + 2)
            stop = length if end == -1 else end + 2
            out.append("\n" * source.count("\n", index, stop))
            index = stop
            continue
        if char in "'\"":
            start = index
            index += 1
            while index < length and source[index] != char:
                index += 2 if source[index] == "\\" else 1
            index += 1
            out.append('""' + "\n" * source.count("\n", start, index))
            continue
        if char == "`":
            index += 1
            while index < length and source[index] != "`":
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == "\n":
                    out.append("\n")
                    index += 1
                    continue
                if source[index:index + 2] == "${":
                    depth, index = 1, index + 2
                    out.append("(")
                    while index < length and depth:
                        if source[index] == "{":
                            depth += 1
                        elif source[index] == "}":
                            depth -= 1
                            if not depth:
                                break
                        out.append(source[index])
                        index += 1
                    out.append(")")
                    index += 1
                    continue
                index += 1
            index += 1
            continue
        if char == "/" and _is_regex_position(out):
            start = index
            index += 1
            in_class = False
            while index < length:
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == "[":
                    in_class = True
                elif source[index] == "]":
                    in_class = False
                elif source[index] == "/" and not in_class:
                    break
                index += 1
            index += 1
            out.append("R" + "\n" * source.count("\n", start, index))
            continue

        out.append(char)
        index += 1
    return "".join(out)


def _is_regex_position(out: list[str]) -> bool:
    """A slash starts a regex only where a value may begin."""
    for char in reversed(out):
        if char in " \t\n":
            continue
        return char in "(,=:[!&|?{};+-*%~^<>" or char == "\n"
    return True


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(WEB)))
def test_delimiters_balance(path: Path):
    code = strip_js(path.read_text())
    stack: list[tuple[str, int]] = []
    line = 1
    for char in code:
        if char == "\n":
            line += 1
        elif char in "([{":
            stack.append((char, line))
        elif char in ")]}":
            assert stack, f"{path.name}: stray '{char}' on line {line}"
            opener, opened = stack.pop()
            assert opener == PAIRS[char], (
                f"{path.name}: '{opener}' opened on line {opened} closed by '{char}' on line {line}"
            )
    assert not stack, f"{path.name}: unclosed {stack[-1][0]} from line {stack[-1][1]}"


def exported_names(path: Path) -> set[str]:
    source = path.read_text()
    names = set(
        re.findall(
            r"^export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)",
            source,
            re.M,
        )
    )
    for group in re.findall(r"^export\s*\{([^}]*)\}", source, re.M):
        for entry in group.split(","):
            entry = entry.strip()
            if entry:
                names.add(entry.split(" as ")[-1].strip())
    return names


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(WEB)))
def test_every_import_resolves_to_a_real_export(path: Path):
    source = path.read_text()
    for names, target in re.findall(
        r"import\s*\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]", source
    ):
        module = (path.parent / target).resolve()
        assert module.is_file(), f"{path.name} imports {target}, which does not exist"
        available = exported_names(module)
        for entry in names.split(","):
            entry = entry.strip().split(" as ")[0].strip()
            if entry:
                assert entry in available, (
                    f"{path.name} imports '{entry}' from {target}, which does not export it"
                )


@pytest.mark.parametrize("path", VIEWS, ids=lambda p: p.stem)
def test_a_view_exposes_the_shell_s_contract(path: Path):
    exports = exported_names(path)
    assert "render" in exports, f"{path.name} must export render(container, ctx)"


@pytest.mark.parametrize("path", VIEWS, ids=lambda p: p.stem)
def test_a_view_that_opens_a_stream_also_tears_it_down(path: Path):
    source = path.read_text()
    if re.search(r"\bstream\s*\(", source) or re.search(r"setInterval\s*\(", source):
        assert "export function dispose" in source, (
            f"{path.name} opens a stream or an interval but never disposes of it, "
            "so switching tabs would leak it"
        )


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(WEB)))
def test_no_interpolated_innerhtml(path: Path):
    for line in path.read_text().splitlines():
        if "innerHTML" in line and ("${" in line or "+" in line):
            pytest.fail(f"{path.name}: innerHTML built from data — use h() instead: {line.strip()}")


# Every helper the toolkit exports. A view that calls one without importing it
# throws at render time and the tab comes up blank — which is exactly what
# happened, and what the import test above could not see, because it only
# checked that the names a file DOES import are real.
TOOLKIT = sorted(exported_names(WEB / "js" / "ui.js") | exported_names(WEB / "js" / "api.js"))


# Anything a browser or the language already provides. Everything else a module
# calls by a bare name has to come from an import or a definition in that file.
JS_GLOBALS = {
    "Array", "Boolean", "Date", "Error", "Intl", "JSON", "Map", "Math", "Number",
    "Object", "Promise", "RegExp", "Set", "String", "Symbol", "WeakMap",
    "AbortController", "Blob", "CustomEvent", "Event", "EventSource", "FormData",
    "Headers", "Image", "IntersectionObserver", "MouseEvent", "MutationObserver",
    "Request", "Response", "ResizeObserver", "TextDecoder", "TextEncoder", "URL",
    "URLSearchParams", "WebSocket", "Worker",
    "addEventListener", "alert", "atob", "btoa", "cancelAnimationFrame", "clearInterval",
    "clearTimeout", "confirm", "decodeURIComponent", "encodeURIComponent", "fetch",
    "isFinite", "isNaN", "parseFloat", "parseInt", "prompt", "queueMicrotask",
    "removeEventListener", "requestAnimationFrame", "setInterval", "setTimeout",
    "structuredClone", "import",
    # keywords and class syntax the regex sees as calls
    "if", "for", "while", "switch", "catch", "return", "typeof", "function", "await",
    "of", "in", "new", "do", "else", "case", "delete", "void", "yield", "async",
    "constructor", "super", "this",
}

# A brace group holding nothing but names, colons, commas and defaults is a
# destructuring pattern or an object literal; either way its names are in scope.
PATTERN_GROUP = re.compile(r"\{([^{};()]*)\}")


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(WEB)))
def test_every_called_name_is_defined_or_imported(path: Path):
    """A call to a helper that no longer exists is a blank page, not a stack trace
    the tests would catch — nothing imports these modules but the browser. This
    walks the bare `name(` calls and insists each one has somewhere to come from."""
    if path.name == "ui.js":
        return
    text = path.read_text()
    source = strip_js(text)

    known = set(JS_GLOBALS)
    for block in re.findall(r"import\s*\{([^}]*)\}", text):
        known |= {n.strip().split(" as ")[-1].strip() for n in block.split(",") if n.strip()}
    known |= set(re.findall(r"import\s+([A-Za-z_$][\w$]*)\s+from", text))
    known |= set(re.findall(r"\b(?:const|let|var|class|function\s*\*?)\s+([A-Za-z_$][\w$]*)", source))
    # Destructured bindings, object shorthand, and every parameter are in scope
    # too — a name that any of those introduces is not a missing helper.
    for group in PATTERN_GROUP.findall(source):
        known |= {n.strip().split(":")[-1].split("=")[0].strip().lstrip(". ")
                  for n in group.split(",")}
    for params in re.findall(r"\(([^()]*)\)", source):
        known |= {p.strip().split("=")[0].strip().lstrip(". ") for p in params.split(",")}
    known |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=>", source))
    # Class and object methods are defined by `name(args) {`, which the call
    # regex also matches; the definition counts as the declaration.
    known |= set(re.findall(r"^\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{",
                            source, re.M))

    called = set(re.findall(r"(?<![.\w$'\"`])([a-z_$][\w$]*)\s*\(", source))
    missing = sorted(name for name in called if name not in known)
    assert not missing, f"{path.name} calls undefined helper(s): {missing}"


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(WEB)))
def test_every_toolkit_helper_used_is_imported(path: Path):
    if path.name in ("ui.js", "api.js"):
        return
    source = strip_js(path.read_text())

    imported: set[str] = set()
    for block in re.findall(r"import\s*\{([^}]*)\}", path.read_text()):
        imported |= {n.strip().split(" as ")[0].strip() for n in block.split(",") if n.strip()}

    # A view may legitimately bind its own `put` or `stat`; only an unbound call
    # is a mistake.
    local = set(re.findall(r"\b(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)", source))
    for params in re.findall(r"function\s+[\w$]*\s*\(([^)]*)\)", source):
        local |= {p.strip().split("=")[0].strip() for p in params.split(",") if p.strip()}
    for params in re.findall(r"\(([^)]*)\)\s*=>", source):
        local |= {p.strip().split("=")[0].strip() for p in params.split(",") if p.strip()}

    for name in TOOLKIT:
        if name in imported or name in local:
            continue
        # A bare call to the helper, not a property access like foo.stat(...).
        if re.search(rf"(?<![.\w]){re.escape(name)}\s*\(", source):
            pytest.fail(f"{path.name} calls {name}() but does not import it from the toolkit")
