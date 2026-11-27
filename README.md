# mutant_rag — historical-bug-based mutation testing for Bitcoin Core

Given a C++ function from Bitcoin Core, `mutant_rag` retrieves similar
**historical bug fixes** from the project's git history. It then asks an LLM (through the
Claude Code, Codex or opencode CLI) to generate **realistic mutants** that re-introduce analogous faults. Each mutant
is a unified diff with an explanation and the historical commits that inspired it.
A separate stage applies, compiles and tests mutants in throwaway git worktrees.

```
git history ──mine──▶ candidates.jsonl ──label──▶ confirmed bugs ──index──▶ FAISS + metadata
                                                                               │
target.cpp ─────────────────────────────────────────────search (top-k)◀────────┘
     │                                                       │
     └──────────────▶ prompt ──agent CLI (no tools)──▶ snippets ──▶ patches (difflib)
                                                                        │
                                       validate: git worktree → apply → build → tests
```

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'      # sentence-transformers, faiss-cpu, numpy, pytest
```

You also need one agent CLI installed and logged in. Pick it with `--backend` on `triage` and
`generate`:

| `--backend` | CLI | `--model` examples | notes |
|---|---|---|---|
| `claude` (default) | [Claude Code](https://docs.claude.com/en/docs/claude-code) | `haiku`, `sonnet`, `opus` | subscription login; `ANTHROPIC_API_KEY` is removed from the child env |
| `codex` | [Codex CLI](https://github.com/openai/codex) | `gpt-5.5` (what your account allows) | ChatGPT login; `OPENAI_API_KEY`/`CODEX_API_KEY` are removed from the child env |
| `opencode` | [opencode](https://opencode.ai) | `openrouter/deepseek/deepseek-v3.2`, `opencode/gpt-5.4-nano` (`opencode models` lists them) | any provider configured in opencode (`opencode auth list`), so cheap and local models work too |

`--model` defaults to the CLI's own default, `--effort` maps to `--effort` (claude),
`model_reasoning_effort` (codex) or `--variant` (opencode), and `--llm-bin` points to a
non-default executable (`--claude-bin` still works).

```bash
python -m mutant_rag generate --backend opencode --model openrouter/deepseek/deepseek-v3.2 \
       --file ~/src/bitcoin/src/txmempool.cpp --function TrimToSize --out mutants.json
python -m mutant_rag triage --backend codex --batch-size 5 --limit 50
```

## Quick start (synthetic data, ~1 minute)

```bash
python -m mutant_rag index --dataset examples/synthetic_bugs.jsonl --index-dir data/demo-index
python -m mutant_rag search   --file examples/target_mempool.cpp --top-k 3 --index-dir data/demo-index
python -m mutant_rag generate --file examples/target_mempool.cpp --top-k 3 --index-dir data/demo-index \
       --out mutants.json
```

`examples/synthetic_bugs.jsonl` contains **hand-written, synthetic** bug/fix pairs. They have fake
`5e5e…` SHAs and `source: "synthetic"`. Use them for demos and tests only, never for research results.

## Real workflow

```bash
# 1. Mine candidate fixes (read-only on the repo). Everything starts as `unreviewed`.
python -m mutant_rag mine --repo ~/src/bitcoin --since 2018-01-01

# 2. Let Claude pre-classify candidates (optional, see "Triage" below)
python -m mutant_rag triage [--limit 50] [--model opus]

# 3. Manually validate candidates (y = confirmed bug fix, n = not, u = uncertain, s = skip, q = quit)
python -m mutant_rag label                              # everything, highest heuristic score first
python -m mutant_rag label --triage-queue uncertain     # only what Claude was unsure about
python -m mutant_rag label --triage-queue audit --sample 30   # blind audit of confident verdicts
python -m mutant_rag label --id 1a2b3c4d5e6f --set confirmed_bug --notes "off-by-one"
python -m mutant_rag label --stats                      # counts + human/Claude agreement

# 4. Index (incremental: only new/changed records are embedded; ineligible ones are pruned)
python -m mutant_rag index                              # human-confirmed only
python -m mutant_rag index --triage-min-confidence 0.8  # + unreviewed records Claude accepts

# 5. Retrieve similar historical bugs for a target function
python -m mutant_rag search --file ~/src/bitcoin/src/txmempool.cpp --function TrimToSize --top-k 5 \
       [--subsystem mempool --subsystem validation] [--exclude-commit <sha>]

# 5b. (Optional) Discover and summarize existing tests covering the target function
python -m mutant_rag discover-tests --file ~/src/bitcoin/src/txmempool.cpp --function TrimToSize \
       [--method agent|static] [--coverage lcov.info] --out test_context.json

# 6. Generate mutants (calls the agent CLI; nothing in the repo is modified)
python -m mutant_rag generate --file ~/src/bitcoin/src/txmempool.cpp --function TrimToSize \
       --top-k 5 --n-mutants 5 [--test-context test_context.json] \
       [--backend opencode --model provider/model] [--dry-run] --out mutants.json

# 7. Validate in temporary worktrees
python -m mutant_rag validate --input mutants.json \
  --build-cmd "cmake -B build -DBUILD_GUI=OFF -DCMAKE_CXX_COMPILER_LAUNCHER=ccache && cmake --build build -j8 --target test_bitcoin" \
 --test-cmd  "build/bin/test_bitcoin --run_test=mempool_tests"
```

`--file` must identify one function. Pass `--function NAME` or `--line N` when the file defines
several. A file with exactly one function, or with no recognizable function at all, is used as-is.
With `--line`, a target too long for the embedder's input is embedded around that line instead of
from its first line (Claude still gets the whole function).

### Mining heuristics: why "fix" is not enough

A commit becomes a candidate only if **all** of these hold:

* Its message scores at least `--min-score` (default 2.0). Positive indicators are weighted
  (crash, overflow, off-by-one, race, UB, CVE, uninitialized, …). Negative indicators in the
  subject subtract (typo, doc/comment, lint, `build:`/`ci:`/`test:` prefixes, refactor, rename,
  scripted-diff, move-only, …). A bare "fix" is worth only +1.
* It is not a merge commit, and it touches ≤ 3 production C++ files with ≤ 60 changed lines.
  Vendored subtrees (leveldb, secp256k1, …), tests and benches don't count as bug locations.
* The change maps to ≤ 3 functions that exist both before and after the commit.

The score and the matched reasons are stored per record (`candidate_score`,
`candidate_reasons`), so the heuristic itself can be audited. Mined records are stored as
**`unreviewed`**, and `index` only includes `confirmed_bug` unless you pass
`--include-unreviewed` (useful for exploration, not for experiments). Re-mining never
overwrites existing records, so manual labels are preserved.

### Triage: Claude-assisted labeling

`triage` sends unreviewed candidates to the selected backend in batches (`--batch-size`, default 5),
using the same tool-less, sandboxed invocation as generation. For each candidate Claude returns:

* `verdict`: `bug_fix`, `not_bug_fix` or `uncertain`, judged against explicit criteria
  (refactors, logging, warnings, test-only changes and features are not bug fixes)
* `confidence` between 0 and 1, a short `reason`, and a `bug_pattern`
* `fix_in_this_function`: whether *this* function's diff contains the fix, or the function only
  changed as plumbing. A real fix commit often touches several functions, and only the ones that
  hold the fix make good retrieval examples.

These go into separate `triage_*` fields. **Triage never writes the human `label`**, so the two
can always be compared. The rules that tie them together:

* `index` ignores triage unless you pass `--triage-min-confidence X`. Then it also admits
  `unreviewed` records with `bug_fix`, confidence ≥ X and `fix_in_this_function` true.
  A human `rejected` always wins, and a later rejection is pruned from the index on the next run.
  `index.json` lists which records were admitted via triage.
* `label --triage-queue uncertain` shows only records that are uncertain, below
  `--min-confidence`, or not yet triaged. Claude's verdict and reason are shown while you review.
* `label --triage-queue audit --sample N` draws a seeded random sample of *confident* verdicts.
  Claude's verdict is **hidden** so it can't anchor your judgment. `label --stats` then reports
  agreement: accuracy, false positives and false negatives. This gives you a measured error rate
  for the triage-admitted part of the dataset.
* The dataset is saved after every batch, so an interrupted run loses nothing. Records that
  already have a verdict are skipped unless `--redo` is given. A failed batch (timeout, CLI
  error) is recorded and the run continues. Each batch's exact prompt and raw response are
  saved under `runs/<id>_triage_*/batches/`.

A suggested protocol for research use: triage everything, hand-label the `uncertain` queue,
audit a random sample of ~30–50 confident verdicts, and report that agreement alongside results.
Optionally, run experiments on both the human-only and human+triage datasets.

## Record format (`data/bugs.jsonl`)

One JSON object per line, one per (commit, file, function):

| field | meaning |
|---|---|
| `id` | `<sha[:12]>:<file>:<function>` |
| `commit_sha`, `parent_sha`, `author_date` | provenance |
| `file_path`, `function_name`, `subsystem` | location (subsystem derived from path) |
| `commit_message` | full message |
| `buggy_code`, `fixed_code` | function before / after the fix |
| `diff` | function-level unified diff |
| `label`, `label_notes`, `reviewer` | `unreviewed` / `confirmed_bug` / `rejected` / `uncertain` |
| `candidate_score`, `candidate_reasons` | mining heuristic output |
| `source`, `tags` | `mined` / `curated` / `synthetic` |
| `triage_verdict`, `triage_confidence`, `triage_reason`, `triage_bug_pattern`, `triage_fix_in_function`, `triage_model`, `triage_run_id` | Claude triage output (empty until triaged) |

You can also curate records by hand. Any JSONL file with these fields can be passed via
`index --dataset` (repeatable).

## Design notes

**Index** (`data/index/`): `index.faiss` is an `IndexIDMap2(IndexFlatIP)` over L2-normalized
vectors, so scores are cosine similarities. `metadata.jsonl` (records and content hashes) and
`manifest.json` (embedder, dim, text-format version) are stored separately. Reloading never
re-embeds. Incremental indexing re-embeds only records whose embedded content changed;
label-only edits update metadata in place. A different embedder or text format is rejected
rather than silently mixed in.

**Embeddings**: the default is `sentence-transformers/all-MiniLM-L6-v2` on CPU, which is
deterministic. Override the model with `--embedder` and the device with `MUTANT_RAG_DEVICE`.
`--embedder hashing-512` is an offline feature-hashing baseline used by tests.

**Focused chunking**: the model reads only 256 tokens, and most Bitcoin Core functions are
longer, so embedding a whole function would keep just its signature and first statements.
Each record's embedded text is therefore built to fit the model's token budget, in this order:
header (subsystem, function, fix subject), then the function's signature plus the lines the fix
changed and as much surrounding context as fits (with `// ...` where lines were left out), then
the fix's removed/added lines, using at most a third of the budget. When the full diff doesn't
fit, only the added lines are kept. Functions that fit are embedded whole. The stored record and
the Claude prompt always use the full function. See `chunking.py`.

**Retrieval** runs in stages: dense top-N → filters (subsystem, label, excluded commits) →
`Reranker`s → top-k. A `Reranker` is any object with
`name` and `rerank(query, candidates) -> candidates` that records its score in
`candidate.scores[name]`. BM25 or AST-similarity rerankers plug into `Retriever(rerankers=[...])`
without changing callers, and every stage's score is saved. Use `--exclude-commit` to hold out
the commit a target was taken from when evaluating.

**LLM backends** (`llm.py`) share one contract: `run_json(prompt, schema, system_prompt)` returns
a schema-valid JSON object or raises a typed `LLMError`. **Claude Code** (`claude_cli.py`) runs:

```
claude -p --output-format json --json-schema <schema> --tools "" --strict-mcp-config \
       --disable-slash-commands --no-session-persistence [--model M] [--effort E]
```

The prompt goes in on stdin, and the CLI runs in a fresh empty temp directory. With no tools,
no MCP and no repo cwd, the model cannot read or modify Bitcoin Core. The schema-validated
object is read from `structured_output`. Non-zero exits, `is_error` results, non-JSON output
and timeouts raise typed errors, and the CLI's stdout/stderr are saved in the run directory.

**Codex** (`codex_cli.py`) runs `codex exec --json --ephemeral --skip-git-repo-check --sandbox
read-only --disable shell_tool -c web_search="disabled" -c mcp_servers={} --output-schema <file> -`
in the same kind of empty temp directory. Codex enforces the schema; the reply is the last
`agent_message` of the JSONL event stream. The system prompt is placed before the user prompt.
A snap-packaged codex has a private `/tmp` and can't read the schema file there: set `TMPDIR` to a
non-hidden directory under your home, or install codex with npm.

**opencode** (`opencode_cli.py`) runs `opencode run --format json --pure --agent mutant-rag
--title mutant-rag [--model provider/model] [--variant E]`. The `mutant-rag` agent is injected
through `OPENCODE_CONFIG_CONTENT` (merged with your config): its prompt is our system prompt,
replacing opencode's long coding-agent prompt, all its tools are disabled and every permission
is denied. opencode has no structured-output flag, so the JSON Schema is appended to the prompt,
and the reply is parsed leniently (code fences and surrounding prose are tolerated) and checked
against the schema (`llm.check_schema`). The per-call cost and token counts opencode reports are
saved in `llm_response.json`. opencode retries provider errors such as rate limits or an
exhausted free quota instead of exiting, so they show up as timeouts; rerun the saved argv with
`--print-logs` to see the cause.

**Patches**: Claude returns `original_snippet` → `mutated_snippet` pairs, not diffs. The tool
locates the snippet inside the target function and builds the unified diff itself with
`difflib` against the exact file bytes. Exact matches are tried first, then a
whitespace-insensitive match with re-indentation. Candidates are rejected and kept with a
`reject_reason` when the snippet is not found or is ambiguous, the change is a no-op or
comment/whitespace-only, it only touches logging/printing statements (`LogPrintf`, `LogDebug`,
`LogInfo`, …, `printf`, `std::cout`/`std::cerr`; log output isn't tested, so these mutants are
noise, and the prompt also tells Claude not to produce them), it exceeds `--max-changed-lines` (default 3), it duplicates another
mutant, or it cites no retrieved historical bug.

**Validation** never touches the main checkout. It creates `git worktree add --detach` worktrees
under a temp directory, runs `git apply --check` and then `git apply`, and runs your build and
test commands (with `{worktree}`, `{file}` and `{repo}` placeholders and `MUTANT_*` env vars). It
then removes the worktrees and prunes. `--worktree-mode shared` (the default) reuses one worktree,
reset between mutants, so incremental/ccache builds stay fast. `per-mutant` gives full
isolation. A **baseline** (unmutated) build and test run is recorded first, so you can confirm
the selected tests pass without mutations. A fingerprint of the main checkout (HEAD and
`git status`) is compared before and after.

Outcomes are `not_applicable`, `applied`, `compile_failed`, `build_timeout`, `compiled`,
`killed`, `test_timeout`, `survived` and `error`. **`survived` is not a verdict.** A survivor
may be equivalent or simply uncovered by the selected tests, and the tool deliberately does
not classify it as useful or non-equivalent.

## Reproducibility

Every command writes `runs/<UTC-timestamp>_<command>_<id>/`:

| file | content |
|---|---|
| `run.json` | argv, parsed args, seed, start/finish time, Python/platform/package versions, git commit of this tool |
| `mining.json` | mining config, stats, repo HEAD, IDs of new records |
| `index.json` | labels indexed, triage threshold, upsert/prune stats, manifest, record IDs, IDs admitted via triage |
| `triage_meta.json`, `batches/*`, `triage_summary.json` | triage config, system prompt, schema, CLI version; per-batch prompt + raw response |
| `target.json` | target file/function/lines, code, sha256, repo base commit, dirty flag |
| `retrieval.json` | index manifest, embedder, filter, ranked results with per-stage scores and SHAs |
| `prompt.txt` | exact prompt sent to the LLM |
| `generation_meta.json` | system prompt, JSON schema, backend config (`llm_config`) and argv, CLI `--version`, models used, prompt sha256 |
| `llm_response.json` | raw CLI response (JSON, or parsed JSONL events for codex/opencode) |
| `mutants.json`, `patches/*.patch` | all candidates, including rejected ones with reasons |
| `validation.json`, `logs/<mutant>/*.log` | per-mutant outcomes, full build/test logs, baseline, checkout fingerprint check |

Seeds (`--seed`, default 1337) are applied to Python, NumPy and torch. None of the agent CLIs expose a
sampling seed, so generation is **not** bit-reproducible. The exact prompt and raw response are
archived instead.

## Tests

```bash
.venv/bin/python -m pytest -q                               # offline: fake claude binary, temp git repos
MUTANT_RAG_LIVE=1 MUTANT_RAG_MODEL=haiku .venv/bin/python -m pytest -q -k live   # one real `claude -p` call
```

`tests/test_e2e.py` runs the whole pipeline through the CLI on the synthetic dataset:
index → incremental re-index → filtered search → generate (fake Claude) → validate. Validate
compiles and runs a test in temporary worktrees and expects one `killed` and one `survived`
(equivalent) mutant.

## Known limitations / next steps

* Function extraction (`cpp.py`) is a heuristic brace matcher, not a parser. It works on single
  historical file snapshots without build flags, which a compiler-based parser can't do well.
  Against libclang on Bitcoin Core (Nov 2026 master, full CMake build), 99.9% of function
  definitions match exactly (same lines). The remaining misses are definitions generated entirely by
  macros, or ones split across `#ifdef` branches. Only one revision was measured, and code outside
  the build (e.g. Qt) wasn't checked. To re-check after changing `cpp.py`, or against a newer
  Bitcoin Core:

  ```bash
  pip install clang   # Python bindings; also needs a libclang library, e.g. `brew install llvm`
  python scripts/check_cpp_extractor.py --repo ~/src/bitcoin --build-dir build --fail-under 99.5
  python scripts/check_cpp_extractor.py --repo ~/src/bitcoin --files 'src/txmempool*'   # quick
  ```

  The build dir needs a `compile_commands.json` (`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`). A full run
  takes a few minutes. Add each new failure the script finds to `tests/test_cpp.py` as a regression
  case.
* Mining heuristics are intentionally conservative and still noisy (expect many `rejected`
  labels). Linking commits to their PRs and labels (e.g. the `Bug` label on GitHub) would be a
  strong additional signal.
* Retrieval is dense-only today. BM25 and AST rerankers are the intended next additions.
* Building Bitcoin Core per mutant is expensive. Use `ccache`, a narrow `--target`, and the
  default shared worktree.
* macOS: `faiss-cpu` and `torch` ship separate OpenMP runtimes that segfault together when
  multithreaded, so the package sets `OMP_NUM_THREADS=1` by default on Darwin. Embedding a large
  dataset is slower as a result, but it is stable and deterministic.

## Layout

```
mutant_rag/
  records.py     BugRecord schema, JSONL I/O, label constants
  cpp.py         C++ function extraction, subsystem mapping
  mining.py      git mining + candidate heuristics
  triage.py      LLM-assisted pre-classification of candidates
  labeling.py    interactive / scripted manual validation, review queues, agreement stats
  chunking.py    focused chunking: narrows long functions to the changed lines for embedding
  embedding.py   sentence-transformers + hashing embedders, token budgets, record/query text
  index.py       FAISS index + metadata + manifest, incremental upsert
  retrieval.py   Retriever, filters, Reranker protocol
  target.py      target function loading (repo/relpath/subsystem detection)
  test_context.py test discovery (read-only agent / static grep / LCOV) & prompt summary
  llm.py         backend-neutral LLM client base, errors, JSON/schema helpers
  claude_cli.py  `claude -p` backend
  codex_cli.py   `codex exec` backend
  opencode_cli.py `opencode run` backend
  generation.py  prompt, JSON schema, snippet→patch post-processing
  validation.py  worktree-based apply/build/test
  provenance.py  run directories, seeds, environment snapshot
  synthetic.py   synthetic dataset (tests/demos)
  cli.py         argparse CLI
tests/           unit + end-to-end tests
scripts/         dev tools (check_cpp_extractor.py: compare cpp.py against libclang)
examples/        synthetic dataset and demo target
```
