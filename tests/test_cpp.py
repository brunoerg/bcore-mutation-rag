from mutant_rag.cpp import (find_functions, function_at_line, function_by_name, strip_comments_and_strings,
                            subsystem_for_path)

SRC = r'''
#include <vector>
#define STACKTOP(i) (stack.at(int64_t{i}))
namespace node {
// A comment with a brace {
static int Free(int a) { return a + 1; }

bool CTxMemPool::Check(const Coins& c) const EXCLUSIVE_LOCKS_REQUIRED(cs)
{
    const char* s = "not a } brace";
    auto f = [&](int x) { return x > 0; };
    if (f(1)) {
        return true;
    }
    return false;
}

class Foo {
public:
    Foo(int x) : m_x{x}, m_y(2) {}
    int Get() const { return m_x; }
    bool operator()(int a) const { return a == m_x; }
    bool operator<(const Foo& o) const { return m_x < o.m_x; }
    std::vector<int> m_v GUARDED_BY(cs){};
private:
    int m_x;
    int m_y;
};
} // namespace node

BOOST_AUTO_TEST_CASE(my_test)
{
    BOOST_CHECK(true);
}
'''


def names(src=SRC):
    return [f.name for f in find_functions(src)]


def test_finds_functions_and_ignores_noise():
    assert names() == ["Free", "CTxMemPool::Check", "Foo", "Get", "operator()", "operator<",
                       "BOOST_AUTO_TEST_CASE(my_test)"]


def test_function_text_and_lines():
    f = function_by_name(SRC, "CTxMemPool::Check")
    text = f.text(SRC)
    assert text.startswith("bool CTxMemPool::Check(")
    assert text.endswith("return false;\n}")
    assert function_at_line(SRC, f.start_line + 3).name == "CTxMemPool::Check"


def test_function_by_unqualified_name():
    assert function_by_name(SRC, "Check").name == "CTxMemPool::Check"
    assert function_by_name(SRC, "Nope") is None


def test_strip_preserves_offsets():
    s = 'int a = 1; // }\n/* { */ const char* x = "}";\n'
    c = strip_comments_and_strings(s)
    assert len(c) == len(s) and c.count("\n") == s.count("\n")
    assert "{" not in c and "}" not in c


def test_line_outside_functions():
    assert function_at_line(SRC, 2) is None


def test_subsystems():
    assert subsystem_for_path("src/wallet/spend.cpp") == "wallet"
    assert subsystem_for_path("src/net_processing.cpp") == "p2p"
    assert subsystem_for_path("src/txmempool.cpp") == "mempool"
    assert subsystem_for_path("src/validation.cpp") == "validation"
    assert subsystem_for_path("/abs/bitcoin/src/script/interpreter.cpp") == "script"
    assert subsystem_for_path("README.md") == "other"


# Regression cases found by scripts/check_cpp_extractor.py (libclang comparison).

def _one(src):
    funcs = find_functions(src)
    assert len(funcs) == 1, [f.name for f in funcs]
    return funcs[0]


def test_subscript_operator_is_not_a_lambda():
    src = "struct V {\n    T& operator[](size_t i) { return d[i]; }\n    const T& operator[](size_t i) const\n    {\n        return d[i];\n    }\n};\n"
    assert names(src) == ["operator[]", "operator[]"]


def test_elaborated_return_types():
    assert names("enum Network ParseNetwork(const std::string& s) {\n    return NET_IPV4;\n}\n") == ["ParseNetwork"]
    assert names("struct timeval MillisToTimeval(int64_t t)\n{\n    return {};\n}\n") == ["MillisToTimeval"]
    assert names("class X { enum VType getType() const { return typ; } };\n") == ["getType"]
    # Still not functions:
    assert names("struct alignas(16) Foo {\n    int x;\n};\nenum class E : uint8_t { A, B };\n") == []


def test_main_function_macro():
    f = _one("#include <x>\n\nMAIN_FUNCTION\n{\n    return 0;\n}\n")
    assert (f.name, f.start_line, f.end_line) == ("main", 3, 6)


def test_conversion_and_literal_operators():
    src = ("struct S {\n"
           "    operator const char*() { return fmt; }\n"
           "    operator std::vector<unsigned char>() const\n    {\n        return {};\n    }\n"
           "    constexpr decltype(auto) operator->() const LIFETIMEBOUND { return get(); }\n"
           "};\n"
           'consteval uint64_t operator""_MiB(unsigned long long v) { return v << 20; }\n')
    assert names(src) == ["operator const char*", "operator std::vector<unsigned char>", "operator->",
                          'operator""_MiB']


def test_paren_groups_before_parameter_list():
    src = ("template <typename T>\n"
           "    requires(!std::same_as<T, int>)\n"
           "constexpr explicit(false) NotNull(T&& u) noexcept : m_ptr(u)\n{\n}\n"
           'extern "C" __attribute__((weak)) void __gcov_reset(void) {}\n'
           "template <typename L>\ndecltype(auto) CustomReadField(L l)\n{\n    return l;\n}\n")
    assert names(src) == ["NotNull", "__gcov_reset", "CustomReadField"]
    assert find_functions(src)[0].start_line == 1  # template header included


def test_concepts_and_requires_expressions_are_not_functions():
    src = ("template<typename T>\n"
           "concept Rng = requires(T& rng) {\n    { rng.rand64() } noexcept -> std::same_as<uint64_t>;\n};\n"
           "template <typename T, typename U>\n"
           "    requires requires(std::ostream& os, const T& t) { os << t; }\n"
           "inline std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p)\n{\n    return os;\n}\n")
    f = _one(src)
    assert (f.name, f.start_line) == ("operator<<", 5)


def test_statement_macros_do_not_glue_onto_next_function():
    src = ("BOOST_AUTO_TEST_SUITE_END()\n\n"
           "BOOST_FIXTURE_TEST_SUITE(coins_tests, BasicTestingSetup)\n\n"
           "static int Helper(int a)\n{\n    return a;\n}\n\n"
           "BOOST_AUTO_TEST_CASE(first)\n{\n    BOOST_CHECK(true);\n}\n\n"
           "BOOST_FIXTURE_TEST_CASE(second,\n                        Setup)\n{\n}\n")
    funcs = find_functions(src)
    assert [(f.name, f.start_line) for f in funcs] == [
        ("Helper", 5), ("BOOST_AUTO_TEST_CASE(first)", 10), ("BOOST_FIXTURE_TEST_CASE(second)", 15)]


def test_lambda_in_constructor_init_list_is_not_a_function():
    src = ("CRPCCommand(std::string category, Fn fn)\n"
           "    : CRPCCommand(category,\n"
           "                  [fn](const Req& r) { return fn().Handle(r); })\n{\n}\n")
    f = _one(src)
    assert (f.name, f.start_line, f.end_line) == ("CRPCCommand", 1, 5)


def test_function_pointer_member_with_brace_init_is_not_a_function():
    assert names("struct F {\n    void (*m_fn)(std::ostream& out){nullptr};\n    int (*m_to)(const void* v){nullptr};\n};\n") == []


def test_header_excludes_leading_comments_and_access_specifiers():
    src = ("class A {\n    int m_x{0}; // trailing comment of the member\npublic:\n"
           "    /** Doc comment. */\n    int Get() const { return m_x; }\n};\n")
    f = _one(src)
    assert f.start_line == 5
    assert f.text(src) == "int Get() const { return m_x; }"


def test_all_caps_function_is_not_named_like_a_test_macro():
    src = "inline uint32_t R11(uint32_t a, uint32_t b) { return a + b; }\nFUZZ_TARGET(addrman)\n{\n}\n"
    assert names(src) == ["R11", "FUZZ_TARGET(addrman)"]
