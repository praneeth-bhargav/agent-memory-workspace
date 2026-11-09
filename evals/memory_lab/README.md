# Memory Lab golden-data draft

These are scenario drafts for the standalone Memory Lab. Some expected behaviors may be labeled while others remain blank. Each file groups one behavior area:

- `memory.json`: memory creation, retrieval, correction, deletion, and ambiguous intent.
- `source_lookup.json`: contact, mail, Slack, calendar, and availability lookups.
- `mock_actions.json`: action-context lookup, mock preparation, confirmation, cancellation, and safety gates.
- `chat_lifecycle.json`: isolated chats, shared long-term memory, pending-action isolation, and reopenable history.
- `handoff.json`: disambiguating two people, confirming a handoff email, and checking the resulting memory state.
- `story_cases.json`: additional cases grounded in the supplied conversation history and mock source fixtures.

The 17 `source_lookup.json` and `mock_actions.json` labels were drafted by the coding agent at the user's request; they are not independent human-reviewed ground truth. Previously human-authored labels remain unchanged. Unlabeled cases are reported as `unlabeled`, and cases whose setup cannot be reproduced are `blocked`. Run the labeled cases locally with `zsh -ic 'export OPENAI_API_KEY; ~/.local/bin/uv run python -m evals.run_memory_lab'`. The isolated runner uses disposable databases and an OpenAI structured-output LLM judge (`EVAL_JUDGE_MODEL` overrides the default judge, `gpt-5.4`; the agent still defaults to `gpt-5.4-mini`); it prints each case's status and duration, then grouped status and score totals; it exits nonzero if any case fails, errors, or is blocked. This command makes paid model calls; do not run it automatically at app startup. No real email, Slack post, or calendar booking is performed.

## LangSmith experiments

1. Add `OPENAI_API_KEY` and `LANGSMITH_API_KEY` to the ignored repository `.env` file (environment variables take precedence). Never paste keys into chat or commit them. Set `LANGSMITH_ENDPOINT` for your account's region and `LANGSMITH_WORKSPACE_ID` if your key requires a workspace.
2. Run one explicitly selected synthetic case:

   ```bash
   ~/.local/bin/uv run python -m evals.run_memory_lab --langsmith --case memory-003 --output /tmp/agent-langsmith.json
   ```

3. Open the experiment URL printed by the SDK or stored as `experiment_url` in the JSON report.

The command uploads the selected synthetic cases, their reference labels (human-authored or agent-drafted as identified above), answers, memory snapshots, tool traces, and traced OpenAI/LangChain model prompts/results. This is an explicit data-export operation. It does not read `memory-lab.sqlite3`, ingest the full history, or enable app tracing. Do not replace the synthetic fixtures with private data without separately reviewing what is uploaded. Local runs disable tracing even if the shell enables it. `evals/synthetic-fixtures.json` fingerprints the approved source files, history, persona, and case files. Changed/missing/extra fixtures block execution; review changes before updating that manifest. Verification captures immutable file bytes, and every trial seeds/reads that snapshot, not files that may change during the experiment.

Datasets are named `agent-memory-lab-<content hash>` so changed case inputs/labels create a separate dataset rather than overwriting a previous version. Example IDs are stable; rerunning an interrupted dataset upload fills missing examples without duplicating them. Manually changed labels/inputs, unexpected examples, or duplicate examples are rejected rather than overwritten. Experiments execute one validated example snapshot, not a later remote re-fetch.

Experiments expose separate `agent_correctness`, `memory_correctness`, and `state_safety` feedback. Inapplicable memory grades and blocked/unlabeled/error cases have **null**, not passing, scores. Labels explicitly mentioning memory/save/forget/remember require a non-null memory judgment; omission becomes an error, not a silent pass. This lexical guard is not a complete semantic applicability classifier. A failed deterministic check blocks both correctness scores; a favorable LLM judge cannot override it. The local report includes counts, scoring denominators, wall-clock case duration, and remaining trials; LangSmith model traces capture provider usage where the SDK supports it. Local reports do not yet aggregate token usage or billing cost.

## Repeat, resume, and coverage

```bash
# Fresh disposable DB for each of three trials; makes paid model calls.
~/.local/bin/uv run python -m evals.run_memory_lab --case memory-003 --repeat 3 --output /tmp/agent-local.json
# Resume the same local report; retry blocked/error trials, retain completed outcomes.
~/.local/bin/uv run python -m evals.run_memory_lab --case memory-003 --repeat 3 --resume --output /tmp/agent-local.json
```

Local resume rejects changed cases, labels, model names, repetitions, runner/memory code, or source fixtures. Hosted runs deliberately start fresh comparison experiments; `--resume --langsmith` is rejected. `--repeat 3` also works for hosted experiments. Output must be a regular `.json` path; an existing file must have the evaluation-report type marker. Databases, symlinks, and unrelated files are rejected. Reports are atomically replaced after every completed case, so completed evidence survives later provider/hosting failures. A later turn/judge error retains already-recorded turn events and states; a hard process kill can still lose the in-flight case. An interrupted in-flight case is retried; model calls already consumed by that case may be billed again.

Human-authored `deterministic_checks` strings remain unchanged. The runner executes the final-state checks in all three handoff cases, including draft state, recipient, send count, memory transfer, and action-context tool order, plus an independent per-turn confirmation gate for email, Slack, and calendar writes. For agent-drafted `source-008`, a deterministic check verifies that at least one explicitly written 24-hour time range is a free 30-minute slot within 11:00–14:00 IST for both Arjun and Rohan and was covered by successful calendar reads or returned availability results; it also checks that no action remains pending. Unrecognized time formats fail closed, and the semantic judge still assesses the complete answer. Every current case has an explicit reproducible setup; setup descriptions are fingerprinted and any change blocks execution until the evaluator setup is reviewed. Unknown future setups or checks still block before model execution rather than being treated as passing.

Blank labels remain visible as unscored coverage gaps; the 17 agent-drafted source-lookup/mock-action labels still need independent review before they can serve as human ground truth. Every user turn must have a label before its conversation is scored. A run containing only unlabeled cases is an inventory, not a passing evaluation, even though it makes no paid calls and exits zero.

Offline verification (no keys or model calls required):

```bash
~/.local/bin/uv run pytest tests/test_langsmith_eval.py tests/test_memory_lab_eval.py
```

This includes a real LangSmith SDK evaluation with `upload_results=False`. A subsequent hosted smoke also verified authentication, dataset/example creation, trace ingestion, and feedback readback for `memory-003`; its report and experiment URL are in `evals/results/langsmith-smoke-2026-11-05.json`. One root run and 13 trace nodes were read back with no root error. This single case is not a full evaluation. The live check also caught and fixed an SDK lifecycle mismatch hidden by an earlier mock: LangSmith 0.14 clients must be closed explicitly rather than used as context managers.

### Observed judge limitation

A real `gpt-5.4-mini` judge accepted a generic clarification for `memory-005` even though the human label explicitly requires naming both preferences. Tightening the rubric alone did not correct that recorded false positive. A `gpt-5.4` judge rejected the same incomplete answer, so it is now the default judge (higher model cost). This is one measured regression, not proof that the stronger judge is infallible. Deterministic checks remain authoritative; human review is still needed for semantic judgments.

A two-case real-model smoke report is retained at `evals/results/local-smoke-2026-11-05.json`: agent 2/2, applicable memory 1/1, confirmation-state checks 2/2. It is a local smoke, not a hosted run or full score. It predates the final immutable-file-snapshot refinement; offline tests subsequently verified that refinement. The no-key inventory reported 28 blocked and 17 unlabeled cases, with no paid calls.
