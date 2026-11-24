from mutant_rag.chunking import ELISION, changed_lines, chunk_code, compact_diff, focus_code, signature_lines
from mutant_rag.embedding import TextBudget, query_text, query_texts, record_text
from mutant_rag.mining import function_diff
from mutant_rag.records import BugRecord


def long_function(n: int = 200, bug_at: int = 150, bug: str = "    if (x < limit) return;") -> str:
    body = [f"    step{i}(x);" for i in range(n)]
    body[bug_at] = bug
    return "bool Check(int x, int limit)\n{\n" + "\n".join(body) + "\n}\n"


def char_budget(n: int) -> TextBudget:
    return TextBudget(len, n)


def test_changed_lines_replacement_and_insertion():
    buggy = "a\nb\nc\nd\n"
    assert changed_lines(buggy, "a\nB\nc\nd\n") == [1]
    assert changed_lines(buggy, "a\nb\nNEW\nc\nd\n") == [1, 2]  # both neighbours of the insertion
    assert changed_lines(buggy, "a\nd\n") == [1, 2]
    assert changed_lines(buggy, buggy) == []


def test_signature_lines():
    assert signature_lines(["int f(int a,", "      int b)", "{", "  x;"]) == 3
    assert signature_lines(["void f() {", "}"]) == 1
    assert signature_lines(["no brace", "here", "at", "all", "x"]) == 1


def test_short_function_is_left_whole():
    code = long_function(5, 2)
    assert focus_code(code, [3], char_budget(10_000).fits) == code


def test_long_function_is_narrowed_around_focus():
    code = long_function()
    bug_line = code.splitlines().index("    if (x < limit) return;")
    out = focus_code(code, [bug_line], char_budget(400).fits)
    assert len(out) <= 400
    lines = out.splitlines()
    assert lines[0] == "bool Check(int x, int limit)" and lines[1] == "{"
    assert "    if (x < limit) return;" in lines
    assert lines[2] == ELISION and lines[-1] == ELISION
    # The window is symmetric around the bug and uses most of the budget.
    i = lines.index("    if (x < limit) return;")
    assert abs((i - 3) - (len(lines) - 2 - i)) <= 1
    assert len(out) > 300


def test_far_apart_focus_lines_get_separate_windows():
    code = long_function(300, 50)
    lines = code.splitlines()
    out = focus_code(code, [52, 250], char_budget(600).fits)
    assert lines[52] in out and lines[250] in out
    assert out.count(ELISION) == 3  # after signature, between windows, at the end
    assert lines[150] not in out.splitlines()


def test_focus_lines_too_big_for_budget_keep_changed_lines():
    code = long_function()
    out = focus_code(code, list(range(10, 150)), char_budget(100).fits)
    assert code.splitlines()[10] in out  # degraded, but still about the change


def test_no_focus_keeps_whole_code():
    code = long_function()
    assert focus_code(code, [], char_budget(100).fits) == code


def test_compact_diff():
    d = function_diff("a\nb\nc\n", "a\nB\nc\n", "f.cpp")
    assert compact_diff(d) == ["-b", "+B"]


def make_record(buggy: str, fixed: str) -> BugRecord:
    return BugRecord(id="x", commit_sha="c" * 40, file_path="src/f.cpp", function_name="Check",
                     commit_message="Fix off-by-one in Check\n\nbody", buggy_code=buggy, fixed_code=fixed,
                     diff=function_diff(buggy, fixed, "src/f.cpp"), subsystem="validation")


def test_record_text_keeps_fix_when_function_is_long():
    buggy = long_function()
    fixed = buggy.replace("x < limit", "x <= limit")
    text = record_text(make_record(buggy, fixed), char_budget(800))
    assert len(text) <= 800
    assert text.startswith("// subsystem: validation\n// function: Check\n// fix: Fix off-by-one in Check\n")
    assert "    if (x < limit) return;" in text  # buggy line in the code window
    assert text.rstrip().endswith("+    if (x <= limit) return;")  # fix diff at the end
    assert "step0(x)" not in text  # the head of the body was dropped, not the fix


def test_record_text_short_function_unchanged_layout():
    buggy = long_function(3, 1)
    fixed = buggy.replace("x < limit", "x <= limit")
    text = record_text(make_record(buggy, fixed), char_budget(10_000))
    assert buggy in text and ELISION not in text


def test_query_text_focus_line():
    code = long_function()
    bug_line = code.splitlines().index("    if (x < limit) return;")
    assert "x < limit" not in query_text(code, "Check", budget=char_budget(400))[:400]
    focused = query_text(code, "Check", focus_line=bug_line, budget=char_budget(400))
    assert len(focused) <= 400 and "x < limit" in focused


def test_record_text_drops_removed_lines_when_diff_is_too_long():
    buggy = long_function(200, 150)
    lines = buggy.splitlines()
    for i in range(140, 160):  # a large rewrite: 20 removed lines before the added one
        lines[i] = f"    legacy_step_with_a_long_name_{i}(x, limit);"
    lines[160] = "    if (x < limit) return;"
    buggy = "\n".join(lines) + "\n"
    fixed = "\n".join(lines[:140] + ["    if (x <= limit) return;"] + lines[161:]) + "\n"
    text = record_text(make_record(buggy, fixed), char_budget(1200))
    diff_part = text.split("// diff:\n", 1)[1]
    assert "+    if (x <= limit) return;" in diff_part
    assert not any(l.startswith("-") for l in diff_part.splitlines())


def test_signature_lines_multiline():
    sig = [
        "template <typename T>",
        "bool Chainstate::ConnectBlock(",
        "    const CBlock& block,",
        "    BlockValidationState& state,",
        "    CBlockIndex* pindex)",
        "    EXCLUSIVE_LOCKS_REQUIRED(cs_main)",
        "{",
        "    return true;",
        "}",
    ]
    assert signature_lines(sig) == 7


def test_chunk_code_short_is_single_chunk():
    code = long_function(5, 2)
    assert chunk_code(code, char_budget(10_000).fits) == [code]


def test_chunk_code_long_covers_all_lines_with_overlap():
    code = long_function(80, 65)
    chunks = chunk_code(code, char_budget(350).fits)
    assert len(chunks) > 2
    for ch in chunks:
        assert len(ch) <= 350
        lines = ch.splitlines()
        assert lines[0] == "bool Check(int x, int limit)" and lines[1] == "{"
    # Every line in the function appears in at least one chunk.
    seen = {l for ch in chunks for l in ch.splitlines()}
    for line in code.splitlines():
        assert line in seen
    # Consecutive chunks overlap on at least one body line.
    for a, b in zip(chunks, chunks[1:]):
        body_a = {l for l in a.splitlines()[2:] if l != ELISION}
        body_b = {l for l in b.splitlines()[2:] if l != ELISION}
        assert body_a & body_b


def test_chunk_code_single_huge_line_makes_progress():
    code = "void f() {\n" + ("    " + "x" * 500 + ";\n") + "    return;\n}\n"
    chunks = chunk_code(code, char_budget(100).fits)
    assert len(chunks) >= 2
    assert any("return;" in ch for ch in chunks)


def test_query_texts_chunks_without_focus_and_focuses_with_line():
    code = long_function(120, 100)
    bug_line = code.splitlines().index("    if (x < limit) return;")
    chunks = query_texts(code, "Check", "validation", budget=char_budget(400))
    assert len(chunks) > 1
    assert all(len(c) <= 400 and c.startswith("// subsystem: validation\n// function: Check\n") for c in chunks)
    assert any("if (x < limit) return;" in c for c in chunks)
    focused = query_texts(code, "Check", "validation", focus_line=bug_line, budget=char_budget(400))
    assert len(focused) == 1 and "if (x < limit) return;" in focused[0]
