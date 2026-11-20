#!/usr/bin/env python3
"""Check mutant_rag.cpp.find_functions against libclang on a configured Bitcoin Core checkout.

This is a development tool, not part of the pipeline. Run it when changing `mutant_rag/cpp.py`, or
now and then against a newer Bitcoin Core, to find C++ shapes the heuristic extractor gets wrong.

libclang parses every translation unit in the build's compile_commands.json with the real flags.
Every outermost function definition with a `{...}` body that it sees in project files under src/
(headers included, deduplicated across TUs) is compared with `find_functions` on the same file.

Requirements (not project dependencies):
    pip install clang          # Python bindings; any recent version works with a newer libclang
    a libclang shared library  # e.g. Homebrew `llvm`, or Debian `libclang-dev`

Usage:
    python scripts/check_cpp_extractor.py --repo ~/src/bitcoin [--build-dir build] [--jobs 8]
        [--files 'src/txmempool*'] [--examples 15] [--json report.json] [--fail-under 99.5]

Outcomes per clang definition: exact (same start and end line), start_differs (same closing brace),
swallowed (inside a larger heuristic range), missed. Per heuristic function: spurious, or
inactive_ifdef (inside a preprocessor branch clang skipped; not an error). Names are compared on the
last `::` component, ignoring template arguments. Definitions without a brace body in the file text
(`= default`, functions generated entirely by a macro) are not counted.
"""
from __future__ import annotations

import argparse
import ctypes
import fnmatch
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mutant_rag.cpp import find_functions, strip_comments_and_strings  # noqa: E402
from mutant_rag.mining import EXCLUDED_PREFIXES, is_production_cpp  # noqa: E402

# Third-party subtrees: never part of what we extract from.
EXTERNAL = ("src/leveldb/", "src/secp256k1/", "src/crc32c/", "src/minisketch/",
            "src/ipc/libmultiprocess/", "src/crypto/ctaes/")
assert all(p in EXCLUDED_PREFIXES for p in EXTERNAL)

_ci = None  # clang.cindex, imported in each process by _setup()
_repo = ""


# ---------------------------------------------------------------------------- libclang side

def find_libclang(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("LIBCLANG_PATH"):
        return os.environ["LIBCLANG_PATH"]
    try:
        libdir = subprocess.run(["llvm-config", "--libdir"], capture_output=True, text=True,
                                check=True).stdout.strip()
        cands = glob.glob(os.path.join(libdir, "libclang.*"))
    except (OSError, subprocess.CalledProcessError):
        cands = []
    cands += glob.glob("/opt/homebrew/opt/llvm/lib/libclang.dylib")
    cands += glob.glob("/usr/local/opt/llvm/lib/libclang.dylib")
    cands += sorted(glob.glob("/usr/lib/llvm-*/lib/libclang.so*"), reverse=True)
    return cands[0] if cands else None  # None: let the bindings search the default paths


def find_resource_dir(libclang: str | None) -> str | None:
    """libclang needs its own builtin headers (stdarg.h, ...) or every TU fails to parse."""
    if not libclang:
        return None
    dirs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.realpath(libclang)), "clang", "*")),
                  key=lambda d: [int(x) for x in re.findall(r"\d+", os.path.basename(d))] or [0])
    return dirs[-1] if dirs else None


def _setup(repo: str, libclang: str | None) -> None:
    global _ci, _repo
    import clang.cindex as ci
    if libclang and not ci.Config.loaded:
        ci.Config.set_library_file(libclang)
    _ci, _repo = ci, repo


def rel(path: str | None) -> str | None:
    if not path:
        return None
    p = os.path.realpath(path)
    if not p.startswith(_repo + "/src/"):
        return None  # generated files under the build dir, system headers
    r = p[len(_repo) + 1:]
    return None if r.startswith(EXTERNAL) else r


def compile_args(entry: dict, resource_dir: str | None) -> list[str]:
    toks = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    out, skip_next = [], False
    for t in toks[1:]:  # drop the compiler itself
        if skip_next:
            skip_next = False
        elif t in ("-o", "-MF", "-MT", "-MQ"):
            skip_next = True
        elif t == "-c" or t == entry["file"] or t.startswith("-M"):
            continue
        else:
            out.append(t)
    return out + (["-resource-dir", resource_dir] if resource_dir else [])


class _SkippedRangeList(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint), ("ranges", ctypes.c_void_p)]


def _skipped_ranges(tu) -> dict[str, list[tuple[int, int]]]:
    """Preprocessor-inactive ranges (clang_getAllSkippedRanges is not in the Python bindings)."""
    lib = _ci.conf.lib
    try:
        fn = lib.clang_getAllSkippedRanges
    except AttributeError:
        return {}
    fn.restype = ctypes.POINTER(_SkippedRangeList)
    fn.argtypes = [_ci.TranslationUnit]
    res = fn(tu)
    arr = ctypes.cast(res.contents.ranges, ctypes.POINTER(_ci.SourceRange))
    out: dict[str, list[tuple[int, int]]] = {}
    for i in range(res.contents.count):
        r = arr[i]
        p = rel(r.start.file.name if r.start.file else None)
        if p:
            out.setdefault(p, []).append((r.start.line, r.end.line))
    lib.clang_disposeSourceRangeList(res)
    return out


def parse_tu(job: tuple[dict, str | None]):
    entry, resource_dir = job
    ci, K = _ci, _ci.CursorKind
    kinds = {K.FUNCTION_DECL, K.CXX_METHOD, K.CONSTRUCTOR, K.DESTRUCTOR, K.CONVERSION_FUNCTION,
             K.FUNCTION_TEMPLATE}
    try:
        tu = ci.Index.create().parse(entry["file"], args=compile_args(entry, resource_dir),
                                     options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    except ci.TranslationUnitLoadError as e:
        return entry["file"], {}, {}, [f"load failed: {e}"]
    errors = [str(d) for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error][:3]
    found: dict[str, set] = defaultdict(set)

    def visit(cur):
        for ch in cur.get_children():
            p = rel(ch.extent.start.file.name if ch.extent.start.file else None)
            if p is None:
                continue  # also prunes whole system headers
            if ch.kind in kinds and ch.is_definition():
                found[p].add((ch.kind.name, ch.spelling, ch.extent.start.line, ch.extent.end.line))
            visit(ch)

    visit(tu.cursor)
    return entry["file"], {p: sorted(v) for p, v in found.items()}, _skipped_ranges(tu), errors


# ---------------------------------------------------------------------------- comparison

def _last_name(name: str) -> str:
    name = re.sub(r"<[^<>]*>", "", name)
    if "operator" in name:
        return "operator" + name.split("operator", 1)[1].replace(" ", "")
    return name.split("::")[-1]


def _names_agree(heur: str, kind: str, clang: str) -> bool:
    if re.fullmatch(r"[A-Z][A-Z0-9_]+(\(.*\))?", heur):
        return True  # test/fuzz macros: the heuristic names them by macro, clang by expansion
    if kind == "CONVERSION_FUNCTION" or re.match(r"operator [A-Za-z_]", clang):
        return re.search(r"\boperator ", heur) is not None  # clang spells the target type its own way
    return _last_name(heur) == _last_name(clang)


def compare_file(path: str, src: str, defs: list, skipped: list, st: Counter, ex: dict) -> None:
    clean = strip_comments_and_strings(src)
    lines, clean_lines = src.split("\n"), clean.split("\n")
    ranges: dict[tuple[int, int], tuple[str, str]] = {}
    for kind, name, s, e in defs:
        ranges.setdefault((s, e), (kind, name))
    bodied = {}
    for (s, e), kn in ranges.items():
        if "{" in "\n".join(clean_lines[s - 1:e]):
            bodied[(s, e)] = kn
        else:
            st["not_counted_no_brace_body"] += 1
    outer = {r: kn for r, kn in bodied.items()
             if not any(o != r and o[0] <= r[0] and r[1] <= o[1] for o in bodied)}

    heur = find_functions(src)
    by_end = defaultdict(list)
    for f in heur:
        by_end[f.end_line].append(f)
    used: set[int] = set()

    for (s, e), (kind, name) in sorted(outer.items()):
        st["clang_defs"] += 1
        cands = [f for f in by_end.get(e, []) if id(f) not in used]
        if cands:
            f = min(cands, key=lambda f: abs(f.start_line - s))
            used.add(id(f))
            if f.start_line == s:
                st["exact"] += 1
            else:
                st["start_differs"] += 1
                ex["start_differs"].append(f"{path}:{s}-{e} clang={name} heuristic={f.name}@{f.start_line}")
            if not _names_agree(f.name, kind, name):
                st["name_differs"] += 1
                ex["name_differs"].append(f"{path}:{s} clang={name} heuristic={f.name}")
            continue
        cover = next((f for f in heur if f.start_line <= s and e <= f.end_line), None)
        if cover:
            st["swallowed"] += 1
            ex["swallowed"].append(f"{path}:{s}-{e} {name} inside heuristic {cover.name}@{cover.start_line}-{cover.end_line}")
        else:
            st["missed"] += 1
            ex["missed"].append(f"{path}:{s}-{e} {kind}:{name} | {lines[s - 1].strip()[:100]}")

    for f in heur:
        if id(f) in used:
            continue
        if any(a <= f.start_line and f.end_line <= b for a, b in skipped):
            st["inactive_ifdef"] += 1
        else:
            st["spurious"] += 1
            ex["spurious"].append(f"{path}:{f.start_line}-{f.end_line} {f.name} | {lines[f.start_line - 1].strip()[:100]}")


# ---------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="Bitcoin Core checkout")
    ap.add_argument("--build-dir", default="build", help="dir with compile_commands.json (relative to --repo)")
    ap.add_argument("--libclang", help="path to libclang.{so,dylib} (default: autodetect, or $LIBCLANG_PATH)")
    ap.add_argument("--resource-dir", help="clang resource dir (default: derived from --libclang)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--files", action="append", default=[],
                    help="only parse TUs and compare files matching this glob, relative to --repo, "
                         "e.g. 'src/txmempool*' (repeatable). Headers are seen through matching TUs.")
    ap.add_argument("--examples", type=int, default=15, help="examples printed per category")
    ap.add_argument("--json", help="write full stats and all examples here")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="exit 1 if the exact-match %% of production code is below this")
    args = ap.parse_args()

    repo = os.path.realpath(os.path.expanduser(args.repo))
    db_path = Path(repo, args.build_dir, "compile_commands.json")
    if not db_path.exists():
        sys.exit(f"{db_path} not found: configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
    try:
        import clang.cindex  # noqa: F401
    except ImportError:
        sys.exit("Python clang bindings missing: pip install clang")
    libclang = find_libclang(args.libclang)
    resource_dir = args.resource_dir or find_resource_dir(libclang)
    print(f"libclang: {libclang or '(bindings default)'}\nresource dir: {resource_dir}", file=sys.stderr)

    _setup(repo, libclang)
    entries = [e for e in json.loads(db_path.read_text()) if rel(e["file"])]
    if args.files:
        entries = [e for e in entries if any(fnmatch.fnmatch(rel(e["file"]), g) for g in args.files)]
    truth: dict[str, set] = defaultdict(set)
    skipped: dict[str, set] = defaultdict(set)
    failed_tus = []
    with Pool(args.jobs, initializer=_setup, initargs=(repo, libclang)) as pool:
        for i, (f, found, sk, errors) in enumerate(
                pool.imap_unordered(parse_tu, [(e, resource_dir) for e in entries]), 1):
            if errors:
                failed_tus.append((rel(f), errors))
            for p, defs in found.items():
                truth[p].update(map(tuple, defs))
            for p, rs in sk.items():
                skipped[p].update(rs)
            if i % 50 == 0 or i == len(entries):
                print(f"parsed {i}/{len(entries)} TUs", file=sys.stderr)
    if failed_tus:
        print(f"\nWARNING: {len(failed_tus)} TUs had parse errors; their results are partial. First:",
              file=sys.stderr)
        for f, errs in failed_tus[:5]:
            print(f"  {f}: {errs[0]}", file=sys.stderr)

    stats = {"production": Counter(), "test": Counter()}
    examples = {"production": defaultdict(list), "test": defaultdict(list)}
    for path in sorted(truth):
        if args.files and not any(fnmatch.fnmatch(path, g) for g in args.files):
            continue
        scope = "production" if is_production_cpp(path) else "test"
        stats[scope]["files"] += 1
        src = Path(repo, path).read_text(errors="replace")
        compare_file(path, src, sorted(truth[path]), sorted(skipped.get(path, [])),
                     stats[scope], examples[scope])

    for scope in ("production", "test"):
        st = stats[scope]
        n = st["clang_defs"] or 1
        print(f"\n== {scope} (mining scope)" if scope == "production" else f"\n== {scope} and other excluded code")
        print(f"   {st['files']} files, {st['clang_defs']} function definitions according to clang")
        for k in ("exact", "start_differs", "swallowed", "missed", "name_differs", "spurious"):
            print(f"   {k:16s} {st[k]:6d}  ({100 * st[k] / n:.2f}%)")
        print(f"   not counted: {st['inactive_ifdef']} heuristic functions in inactive #if branches, "
              f"{st['not_counted_no_brace_body']} clang definitions without a brace body")
        for k, v in sorted(examples[scope].items()):
            print(f"\n   -- {scope}:{k} ({len(v)})")
            for x in v[:args.examples]:
                print(f"      {x}")
            if len(v) > args.examples:
                print(f"      ... {len(v) - args.examples} more")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"repo_head": subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip(),
             "stats": {k: dict(v) for k, v in stats.items()},
             "examples": {k: dict(v) for k, v in examples.items()},
             "failed_tus": failed_tus}, indent=1))

    prod = stats["production"]
    pct = 100 * prod["exact"] / (prod["clang_defs"] or 1)
    if args.fail_under is not None and pct < args.fail_under:
        print(f"\nFAIL: production exact match {pct:.2f}% < {args.fail_under}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
