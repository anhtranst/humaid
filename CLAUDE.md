# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zero-shot tweet classification for humanitarian response categories (HumAID labels) using OpenAI's Chat Completions with Structured Outputs and the Batch API. The system classifies tweets into 10 humanitarian-assistance categories without fine-tuning, using dynamic prompting rules.

Authors: Anh Tran, Hongmin Li. License: MIT.

## Setup

```bash
python -m venv humaid-env
source humaid-env/bin/activate  # Windows: humaid-env\Scripts\activate
pip install -r requirements.txt
```

API keys go in `.env` (never committed):
```
OPENAI_API_KEY_1=sk-...
OPENAI_API_KEY_2=sk-...
OPENAI_API_KEY=${OPENAI_API_KEY_1}
```

## Architecture

### Package: `humaidclf/`

The core Python package with the full classification pipeline:

- **io.py** — TSV loading (`load_tsv`), run directory creation (`plan_run_dirs`)
- **prompts.py** — `LABELS` (canonical order), `SYSTEM_PROMPT`, `make_user_message()`. Label order is enforced everywhere (enum, charts, reports)
- **batch.py** — OpenAI Batch API integration: sync dry-run, request building, batch submission/polling/download/parsing, API key switching (global and context-manager)
- **budget.py** — Token estimation per request/dataset, token index building, sharding large datasets by token budget
- **eval.py** — Macro-F1 (primary metric), confusion matrices, per-class metrics, mistake analysis, chart generation
- **report.py** — Curated results dashboard: `promote_run_to_results()` copies runs, `build_results_index()` generates `results/index.html`
- **runner.py** — `run_experiment()` (end-to-end) and `resume_experiment()` (from `batch_meta.json`)
- **runner_sharded.py** — Sharded orchestration for large datasets
- **stratify.py** — Stratified k-fold splitting preserving class ratios

### Rules: `rules/`

Kept outside the package for fast iteration. Contains evolving prompt rule variants:
- `RULES_BASELINE` — Original HumAID definitions
- `RULES_1` — Compact single-line (cost-optimized)
- `RULES_2` — Medium-detail with "pick PRIMARY INTENT" framing
- `RULES_3` — Multi-line with Definition/Include/Exclude guidance
- `RULES_4` — Further expansion

Access via `from rules import RULES_1` or `get_rule("RULES_1")`.

### Data: `Dataset/HumAID/`

10 disaster events, each with `<event>_train.tsv`, `<event>_dev.tsv`, `<event>_test.tsv`. TSV columns: `tweet_id`, `tweet_text`, `class_label`.

### Output directories

- **`runs/`** — Auto-generated: `<event>/<split>/<model>/<timestamp-tag>/` containing `requests.jsonl`, `outputs.jsonl`, `predictions.csv`, `batch_meta.json`, and `analysis/` (charts, metrics, mistakes)
- **`results/`** — Manually curated best runs for presentation. `results/index.html` is the generated dashboard

### Notebooks

- `00_build_results_index.ipynb` — Rebuild `results/index.html`
- `01-14_zeroshot_*.ipynb` — Experiment notebooks (naming: `NN_zeroshot_<model>_humaid_<split>_<rules>.ipynb`)
- `zz_dataset_explorer.ipynb` — Dataset visualization
- `zz_sanity_test.ipynb`, `zz_test.ipynb` — Testing/validation

## Key Design Patterns

**Dynamic schema generation**: Structured Output JSON schema is generated per-event, restricting predictions to only labels present in ground truth. This prevents hallucinated labels.

**Truth-only scope for evaluation**: Metrics only include labels present in ground truth, so events missing certain classes aren't penalized.

**Model family detection**: GPT-5*/O3*/O4* models use `max_completion_tokens`; older models use `max_tokens`. This is handled automatically in `batch.py`.

**Resilient batch pipeline**: Preflight probe validates API/model/schema before submission. Failed/missing predictions get a synchronous patch pass. Batch metadata is checkpointed for resumption.

**API key switching**: Multiple keys in `.env`, switchable globally (`set_api_key_env`) or per-block (`with use_api_key_env("OPENAI_API_KEY_2")`). Used to route datasets to different API tier limits.

## Common Workflow

```python
from dotenv import load_dotenv; load_dotenv()
from rules import RULES_1
from humaidclf import run_experiment

plan, preds, summary = run_experiment(
    dataset_path="Dataset/HumAID/<event>/<event>_<split>.tsv",
    rules=RULES_1,
    model="gpt-4o-mini",
    tag="modeS-RULES1",
    dryrun_n=20,
    poll_secs=60,
    do_analysis=True,
)
```

Resume an interrupted run: `resume_experiment(plan["dir"])`

Rebuild results dashboard: `build_results_index("results", out_html="results/index.html")`

## 10 Canonical Labels (order enforced in code)

```
caution_and_advice
displaced_people_and_evacuations
infrastructure_and_utility_damage
injured_or_dead_people
missing_or_found_people
requests_or_urgent_needs
rescue_volunteering_or_donation_effort
sympathy_and_support
other_relevant_information
not_humanitarian
```
