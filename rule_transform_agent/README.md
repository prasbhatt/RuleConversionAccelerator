# Business Rule Transformation Agent — Starter Scaffold

This is a working starter project for the RAG-based, agentic business-rule
transformation system described in the thesis blueprint. It is designed to
be opened directly in VS Code and built up incrementally.

## What already works, right now, with no API key

Everything in `src/parser.py`, `src/schema.py` and `src/interpreter.py` is
**pure, deterministic Python with no external dependencies**. It has its own
test suite and requires no OpenAI credits and no AWS account to try:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

You should see 6 passing tests. These prove, independently of any LLM, that:
- the deterministic parser correctly turns a raw source rule into an
  ordered-program AST,
- the AST validates against the canonical schema v2.0,
- the mini-interpreter actually *executes* that AST and reproduces the
  exact behaviour of the worked example from the thesis
  (`Org_PIM_11361_Fire-RatedDownlights_Bullets_03 COMPLEX`).

This deterministic core is intentionally built and proven correct **before**
any LLM is introduced, because it is the ground everything else stands on:
the agent's generation prompt is grounded by this parser's draft output, and
the evaluation harness's "functional match" metric is powered by this same
interpreter.

## What needs your OpenAI key and dataset to run

1. Create a `.env` file in the project root and set your API key and workbook
    path. For the workbook currently in `data/`, use:

    ```dotenv
    OPENAI_API_KEY=your_openai_api_key
    GROUND_TRUTH_XLSX=./data/Anonymised_Executable_Rule_Ground_Truth_Dataset 5.xlsx
    ```

    `.env` is excluded from Git; do not commit API keys.
2. The ground-truth workbook must contain the `Synthetic Rules` sheet. If you
    use a different workbook, set `GROUND_TRUTH_XLSX` to its path instead.
3. Build the local vector index (one-time, or whenever the dataset changes):

    ```powershell
    python -m src.cli build-index
    ```

4. Convert a brand-new source rule you have never shown the model before:

    ```powershell
    python -m src.cli convert --rule-file ".\rule.txt" --name "Org_PIM_99999_Some_New_Rule" --target-attribute "Bullet Point 05" --target-locale "en_GB" > ".\converted_rule.json"
    ```

    Replace `rule.txt` with the path to a text file containing the source rule.
    The converted JSON is written to `converted_rule.json`; status messages
    remain in the terminal.
5. Once you are happy with prompt/retrieval tuning (done by hand, or against
    the Validation split), run the held-out evaluation:

    ```powershell
    python -m src.cli evaluate
    ```

   This prints `schema_valid_rate`, `variable_safety_rate` and
   `functional_match_rate` computed over the untouched Test split — the
   headline numbers for the thesis evaluation chapter.

## Project layout

```
rule_transform_agent/
├── requirements.txt
├── .env.example
├── data/                          <- put the ground-truth .xlsx here
├── src/
│   ├── config.py                  <- one place for all settings
│   ├── schema.py                  <- canonical JSON Schema v2.0 + validators
│   ├── interpreter.py              <- executes the ordered-program AST
│   ├── parser.py                   <- source rule text -> draft AST
│   ├── dataset.py                  <- loads the ground-truth workbook
│   ├── embeddings.py                <- OpenAI embeddings + local cache
│   ├── vector_store.py              <- Chroma wrapper (swap for AWS later)
│   ├── evaluate.py                  <- Test-split evaluation harness
│   ├── cli.py                       <- `python -m src.cli ...`
│   └── agent/
│       ├── state.py                 <- shared AgentState
│       ├── prompts.py                <- classify/generate/repair prompts
│       ├── nodes.py                  <- 8 node functions (framework-free)
│       └── graph.py                  <- wires nodes into a LangGraph graph
└── tests/
    └── test_schema.py                <- parser + schema + interpreter tests
```

## Design notes worth knowing before you extend this

- **Every node function in `agent/nodes.py` is plain Python.** Only
  `agent/graph.py` imports LangGraph. This means you can unit-test or even
  swap the orchestration framework later without rewriting the actual agent
  logic — a defensible choice to describe and justify in the thesis.
- **The repair loop is capped** by `MAX_REPAIR_ATTEMPTS` in `.env`. If a rule
  still fails validation after that many attempts, it is returned anyway but
  with `requires_human_review = True` rather than silently discarded or
  passed off as trustworthy — mirroring the `requiresHumanReview` flag
  already present in the ground-truth dataset for Complex rules.
- **Cost control:** `classify_node` deliberately uses the cheaper
  `CLASSIFICATION_MODEL`, while only `generate_node` / `repair_node` use the
  more capable (and more expensive) `GENERATION_MODEL`. The embedding cache
  in `embeddings.py` means the Train partition is only ever embedded once.
- **`vector_store.py` is intentionally swappable.** The `AWSVectorStoreAdapter`
  stub documents exactly what changes when you move from the local Chroma
  prototype to the AWS production path (Amazon S3 Vectors behind a Bedrock
  Knowledge Base) — the `query()` interface used by the rest of the agent
  does not change.
