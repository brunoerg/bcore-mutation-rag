"""Lightweight, dependency-free C++ helpers.

Function extraction here is a heuristic brace matcher, not a parser. It handles
the common Bitcoin Core shapes (free functions, qualified member functions,
in-class member definitions, constructor initializer lists, trailing
annotations such as EXCLUSIVE_LOCKS_REQUIRED), which is enough to slice
historical fixes down to function level. It needs no build flags, so it also works
on historical file snapshots from `git show`.

`scripts/check_cpp_extractor.py` measures it against libclang on a configured
Bitcoin Core build. Run it after changing this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NON_FUNCTION_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "do", "else", "try", "return",
    "namespace", "class", "struct", "union", "enum", "sizeof", "decltype",
    "static_assert", "alignas", "alignof", "noexcept",
}
# Parenthesized groups that can precede a function's parameter list and must be skipped:
# `decltype(auto) f()`, `requires(C<T>) X()`, `explicit(false) X()`, `__attribute__((weak)) void f()`.
_SKIP_PAREN_GROUPS = {"decltype", "requires", "explicit", "alignas", "__attribute__", "__declspec"}
# `enum Network ParseNetwork(...)`, `struct timeval f(...)`: elaborated return types.
_ELABORATED_RETURN = re.compile(
    r"\s*(?:enum|struct|class|union)\s+(?:class\s+|struct\s+)?[A-Za-z_][\w:]*(?:<[^<>()]*>)?[\s*&]+[A-Za-z_~][\w:]*\s*\(")
# Brace-bodied macros with no parameter list, mapped to the function they define.
_BODY_MACROS = {"MAIN_FUNCTION": "main"}
# A statement-like macro invocation on its own line(s), e.g. `BOOST_AUTO_TEST_SUITE(x)`.
_STATEMENT_MACRO = re.compile(r"\s*[A-Z][A-Z0-9_]+\s*\(")
_ACCESS_SPECIFIER = re.compile(r"(public|private|protected)\s*:")
# Clang thread-safety / attribute macros that precede brace-initialized members.
_ANNOTATION_MACROS = {
    "GUARDED_BY", "PT_GUARDED_BY", "EXCLUSIVE_LOCKS_REQUIRED", "SHARED_LOCKS_REQUIRED",
    "LOCKS_EXCLUDED", "ACQUIRED_AFTER", "ACQUIRED_BEFORE", "LOCK_RETURNED",
    "EXCLUSIVE_LOCK_FUNCTION", "UNLOCK_FUNCTION", "LIFETIMEBOUND", "ALIGNAS",
}
_IDENT = re.compile(r"[A-Za-z_~][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CppFunction:
    name: str  # possibly qualified, e.g. "CTxMemPool::check"
    start: int  # char offset of the header start (inclusive)
    end: int  # char offset just past the closing brace
    start_line: int  # 1-based
    end_line: int  # 1-based, inclusive

    def text(self, source: str) -> str:
        return source[self.start:self.end]


def strip_comments_and_strings(src: str) -> str:
    """Blank out comments and string/char literals, preserving offsets and newlines."""
    out = list(src)
    i, n = 0, len(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if c == "#" and src[src.rfind("\n", 0, i) + 1 : i].strip() == "":
            # Preprocessor directive, including backslash-continued lines.
            j = i
            while True:
                nl = src.find("\n", j)
                if nl == -1:
                    j = n
                    break
                if src[nl - 1] == "\\":
                    j = nl + 1
                    continue
                j = nl
                break
            blank(i, j)
            i = j
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
        elif c == "R" and src.startswith('R"', i) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            m = re.match(r'R"([^()\\ ]{0,16})\(', src[i:])
            if m:
                close = ")" + m.group(1) + '"'
                j = src.find(close, i + m.end())
                j = n if j == -1 else j + len(close)
                blank(i, j)
                i = j
            else:
                i += 1
        elif c in "\"'":
            # Skip digit separators like 1'000'000.
            if c == "'" and i > 0 and src[i - 1].isalnum() and i + 1 < n and src[i + 1].isalnum():
                i += 1
                continue
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            blank(i, j)
            i = j
        else:
            i += 1
    return "".join(out)


def _match_braces(clean: str) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for i, c in enumerate(clean):
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            pairs[stack.pop()] = i
    return pairs


def _header_start(clean: str, open_idx: int, close_of: dict[int, int]) -> int:
    """Walk back from a '{' to the start of its header, skipping balanced (...) and
    brace-initializers like `m_x{y}` that appear in constructor init lists."""
    open_of = {v: k for k, v in close_of.items()}
    i = open_idx - 1
    paren = 0
    while i >= 0:
        c = clean[i]
        if c == ")":
            paren += 1
        elif c == "(":
            if paren == 0:
                return i + 1
            paren -= 1
        elif paren == 0:
            if c == ";" or c == "{":
                return i + 1
            if c == "}":
                o = open_of.get(i)
                # Brace-init in a constructor init list (`m_x{y}`): identifier or '>'
                # directly before the '{'. Skip over it and keep walking back.
                # Require no whitespace before '{' so that `) const\n{` bodies are never skipped.
                if o is not None and o > 0 and (clean[o - 1].isalnum() or clean[o - 1] in "_>"):
                    i = o - 1
                    continue
                # Body of a requires-expression in a template's requires-clause:
                # `requires requires(T t) { t.f(); }`. Also part of the header.
                if o is not None and _is_requires_body(clean, o):
                    i = o - 1
                    continue
                return i + 1
            if c == "#":  # preprocessor line
                nl = clean.find("\n", i)
                return nl + 1 if nl != -1 else i + 1
        i -= 1
    return 0


def _is_requires_body(clean: str, open_idx: int) -> bool:
    """True if the '{' at open_idx opens a requires-expression: `requires {` or `requires(...) {`."""
    j = open_idx - 1
    while j >= 0 and clean[j].isspace():
        j -= 1
    if j >= 0 and clean[j] == ")":
        depth = 0
        while j >= 0:
            depth += {")": 1, "(": -1}.get(clean[j], 0)
            if depth == 0:
                break
            j -= 1
        j -= 1
    return re.search(r"\brequires\s*$", clean[max(0, j - 40):j + 1]) is not None


def _function_name(header: str) -> str | None:
    """Return the function name for a header that looks like a function definition."""
    h = header.strip()
    if h in _BODY_MACROS:
        return _BODY_MACROS[h]
    if not h or "(" not in h:
        return None
    # Drop template<...> prefixes and access specifiers.
    h = re.sub(r"^\s*(public|private|protected)\s*:", "", h)
    # Drop brace groups (requires-expression bodies, init-list brace-inits): their `<<`
    # and `<` would otherwise be counted as template angle brackets.
    while (stripped := re.sub(r"\{[^{}]*\}", " ", h)) != h:
        h = stripped
    first_word = re.match(r"\s*([A-Za-z_]\w*)", h)
    if first_word and first_word.group(1) in _NON_FUNCTION_KEYWORDS - _SKIP_PAREN_GROUPS:
        if not _ELABORATED_RETURN.match(h):
            return None
    pre_paren = h.split("(", 1)[0]
    if re.search(r"\]\s*$", pre_paren) and not re.search(r"\boperator\s*\[\s*\]\s*$", pre_paren):
        return None  # lambda: `[captures](...)`
    if h.endswith("="):
        return None  # aggregate initializer
    # Find the first top-level '(' whose preceding token is an identifier (or operator).
    # Mask operator symbols (operator<, operator->, ...) so they don't count as angle brackets.
    masked = re.sub(
        r"operator\s*(<=>|<<=|>>=|<<|>>|<=|>=|->|<|>)",
        lambda m: m.group(0)[: -len(m.group(1))] + "@" * len(m.group(1)),
        h,
    )
    depth_angle = 0
    depth_paren = 0
    for m in re.finditer(r"[<>()]", masked):
        ch = m.group()
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif depth_paren:
            if ch == "(":
                depth_paren += 1
        elif ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle = max(0, depth_angle - 1)
        elif ch == "(" and depth_angle == 0:
            before = h[: m.start()].rstrip()
            if before.endswith("operator") and h[m.start() : m.start() + 2] == "()":
                depth_paren += 1
                continue  # `operator()(...)`: the parameter list is the next paren group
            op = re.search(r"(\boperator\s*(\(\)|\[\]|[^\s\w(]+|\s+[^()]+))$", before)
            if op:
                name_end = op.start()
                prefix = re.search(r"([\w:~]*)$", before[:name_end])
                name = re.sub(r"\s+", " ", (prefix.group(1) if prefix else "") + op.group(1)).strip()
                # User-defined literal: `operator""_MiB` (the "" was blanked with the strings).
                return re.sub(r"\boperator (_\w+)$", r'operator""\1', name)
            ident = re.search(r"((?:[A-Za-z_]\w*(?:<[^<>()]*>)?::)*~?[A-Za-z_]\w*)$", before)
            if not ident:
                return None
            name = ident.group(1)
            if name in _SKIP_PAREN_GROUPS:
                depth_paren += 1
                continue
            if re.match(r"\(\s*[*&]", h[m.start():]):
                return None  # function-pointer member: `void (*m_fn)(int){nullptr};`
            last = name.split("::")[-1]
            if last in _NON_FUNCTION_KEYWORDS or last in _ANNOTATION_MACROS:
                return None
            # Test macros (BOOST_AUTO_TEST_CASE(x), FUZZ_TARGET(x)): keep the first
            # argument so that each test case gets a distinct name.
            if re.fullmatch(r"[A-Z][A-Z0-9_]+", name) and before.strip() == name:
                arg = re.match(r"\(\s*([\w:]+)", h[m.start():])
                if arg:
                    return f"{name}({arg.group(1)})"
            return name
    return None


def _skip_statement_macros(clean: str, hs: int, open_idx: int) -> int:
    """Skip macro invocations with no trailing ';' that sit on their own line(s) before the
    real header, e.g. `BOOST_FIXTURE_TEST_SUITE(x, Setup)` before the first test case. A macro
    that is itself the header (`BOOST_AUTO_TEST_CASE(x)` directly followed by `{`) is kept."""
    while m := _STATEMENT_MACRO.match(clean, hs, open_idx):
        depth, j = 1, m.end()
        while j < open_idx and depth:
            depth += {"(": 1, ")": -1}.get(clean[j], 0)
            j += 1
        nl = re.compile(r"[ \t]*\n").match(clean, j, open_idx)
        if depth or not nl or not clean[nl.end():open_idx].strip():
            break
        hs = nl.end()
    return hs


def _trim_header_start(clean: str, hs: int, open_idx: int) -> int:
    """Skip leading whitespace, comments, preprocessor lines and access specifiers."""
    while True:
        while hs < open_idx and clean[hs].isspace():
            hs += 1
        m = _ACCESS_SPECIFIER.match(clean, hs, open_idx)
        if not m:
            return hs
        hs = m.end()


def find_functions(src: str) -> list[CppFunction]:
    """Return function definitions (outermost function-like blocks) in source order."""
    clean = strip_comments_and_strings(src)
    close_of = _match_braces(clean)
    funcs: list[CppFunction] = []
    covered_until = -1
    for o in sorted(close_of):
        if o < covered_until:
            continue  # nested inside an accepted function (lambdas, blocks)
        if o > 0 and (clean[o - 1].isalnum() or clean[o - 1] in "_>"):
            continue  # brace-init such as `m_x{y}` in a constructor init list
        hs = _header_start(clean, o, close_of)
        if hs > 0 and clean[hs - 1] == "(":
            continue  # inside an argument list, e.g. a lambda passed in a constructor init list
        hs = _skip_statement_macros(clean, hs, o)
        name = _function_name(clean[hs:o])
        if not name:
            continue
        c = close_of[o]
        hs = _trim_header_start(clean, hs, o)
        start_line = src.count("\n", 0, hs) + 1
        end_line = src.count("\n", 0, c) + 1
        funcs.append(CppFunction(name, hs, c + 1, start_line, end_line))
        covered_until = c
    return funcs


def function_at_line(src: str, line: int) -> CppFunction | None:
    for f in find_functions(src):
        if f.start_line <= line <= f.end_line:
            return f
    return None


def function_by_name(src: str, name: str) -> CppFunction | None:
    """Match a full or unqualified name; the first definition wins."""
    funcs = find_functions(src)
    for f in funcs:
        if f.name == name:
            return f
    for f in funcs:
        if f.name.split("::")[-1] == name.split("::")[-1]:
            return f
    return None


# Ordered: first matching prefix wins.
_SUBSYSTEM_RULES: list[tuple[str, str]] = [
    ("src/wallet/", "wallet"),
    ("src/qt/", "gui"),
    ("src/rpc/", "rpc"),
    ("src/script/", "script"),
    ("src/consensus/", "consensus"),
    ("src/policy/", "policy"),
    ("src/node/", "node"),
    ("src/kernel/", "kernel"),
    ("src/index/", "index"),
    ("src/crypto/", "crypto"),
    ("src/util/", "util"),
    ("src/primitives/", "primitives"),
    ("src/interfaces/", "interfaces"),
    ("src/ipc/", "ipc"),
    ("src/zmq/", "zmq"),
    ("src/test/", "test"),
    ("src/bench/", "bench"),
    ("src/net_processing", "p2p"),
    ("src/net", "p2p"),
    ("src/addrman", "p2p"),
    ("src/netaddress", "p2p"),
    ("src/protocol", "p2p"),
    ("src/headerssync", "p2p"),
    ("src/txmempool", "mempool"),
    ("src/txorphanage", "mempool"),
    ("src/validation", "validation"),
    ("src/chain", "validation"),
    ("src/coins", "validation"),
    ("src/txdb", "validation"),
    ("src/pow", "consensus"),
    ("src/dbwrapper", "storage"),
    ("src/flatfile", "storage"),
    ("src/init", "init"),
    ("src/common/", "common"),
    ("src/serialize", "serialization"),
    ("src/streams", "serialization"),
    ("src/http", "rpc"),
    ("src/bitcoin-cli", "cli"),
    ("src/psbt", "psbt"),
    ("src/key", "keys"),
    ("src/pubkey", "keys"),
]


def subsystem_for_path(path: str) -> str:
    p = path.replace("\\", "/")
    if not p.startswith("src/") and "/src/" in p:
        p = "src/" + p.split("/src/", 1)[1]
    for prefix, name in _SUBSYSTEM_RULES:
        if p.startswith(prefix):
            return name
    return "other"


def is_cpp_source(path: str) -> bool:
    return path.endswith((".cpp", ".cc", ".cxx", ".h", ".hpp"))
