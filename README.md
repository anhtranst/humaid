# HumAID Tweet Classification (Zero-shot with OpenAI Batch)

Zero-shot tweet classification for humanitarian response categories (HumAID-style labels) using OpenAI's **Chat Completions + Structured Outputs** and the **Batch API**.  
Outputs include predictions and analysis artifacts (confusion matrices, per-class metrics, mistakes).

## Labels

```
caution_and_advice
sympathy_and_support
requests_or_urgent_needs
displaced_people_and_evacuations
injured_or_dead_people
missing_or_found_people
infrastructure_and_utility_damage
rescue_volunteering_or_donation_effort
other_relevant_information
not_humanitarian
```

## Quick Start

### 1) Requirements

- Python 3.10+
- Packages: `pandas`, `numpy`, `matplotlib`, `requests`, `python-dotenv`
- An OpenAI API key (models: `gpt-4o-mini`, `gpt-4o` )

```bash
pip install -r requirements.txt
# or:
pip install pandas numpy matplotlib requests python-dotenv
```

### 2) Configure your API key (do NOT commit it)

Create a `.env` file in the repo root:

```
OPENAI_API_KEY= [YOUR-KEY]
```

`.gitignore` already ignores `.env`. If a key was ever committed, rotate it and purge from history (e.g., `git filter-repo`).

### 3) Project layout

```
Dataset/
  HumAID/<event>/<event>_<split>.tsv     # Input TSVs
humaidclf/
  __init__.py
  io.py            # load_tsv, plan_run_dirs
  prompts.py       # LABELS (ordered as above), SYSTEM_PROMPT, make_user_message
  batch.py         # sync_test_sample, build_requests_jsonl_S, batch helpers, parser
  eval.py          # metrics + analysis & plots
runs/
  <event>/<split>/<model>/<timestamp>-<tag>/
    requests.jsonl
    outputs.jsonl
    predictions.csv
    analysis/                      # per-run analysis lives here
      mistakes.csv
      charts/
        confusion_matrix_counts.png
        confusion_matrix_row_normalized.png
        confusion_matrix.csv
        confusion_matrix_row_normalized.csv
        per_class_f1.png
        per_class_error_rate.png
        top_confusions.png
        per_class_metrics.csv
        summary.json
```

We intentionally store analysis inside each run folder so repeated analyses don't mix across runs.

---

## How to Use

Examples below assume you're in a Jupyter notebook or a Python script at the repo root.

### A) Load data (TSV)

Expected columns: `tweet_id`, `tweet_text`, and (optional) `class_label` for ground truth.

```python
from dotenv import load_dotenv; load_dotenv()
from humaidclf import load_tsv

dataset_path = "Dataset/HumAID/california_wildfires_2018/california_wildfires_2018_train.tsv"
df = load_tsv(dataset_path, id_col="tweet_id", text_col="tweet_text", label_col="class_label")
len(df), df.head(2)
```

### B) Choose rules (zero-shot)

Rules are short text you supply. Swap them to A/B performance vs token cost.

```python
RULES = '''
Pick ONE label for the tweet's PRIMARY INTENT.
- caution_and_advice: warnings/instructions
- sympathy_and_support: prayers/condolences/praise (no logistics)
- requests_or_urgent_needs: asking for help/supplies/services
- displaced_people_and_evacuations: evacuation/relocation/shelter
- injured_or_dead_people: injuries/casualties/deaths
- missing_or_found_people: explicit missing or found/reunited
- infrastructure_and_utility_damage: asset damage/outages CAUSED BY the disaster
- rescue_volunteering_or_donation_effort: offering help; organizing rescues/donations/volunteers/events
- other_relevant_information: on-topic facts/stats/updates when none above fits (event/hashtag/location+disaster term or official update = on-topic)
- not_humanitarian: unrelated or no clear disaster context
Return only the label.
'''
```

### C) Dry-run a small sample (sanity check)

```python
from humaidclf import sync_test_sample, macro_f1
demo = sync_test_sample(df, n=20, rules=RULES, model="gpt-4o-mini", temperature=0.0, seed=42)
print("Sample Macro-F1:", macro_f1(demo))
demo.head()
```

### D) Plan output directories & build the Batch requests

```python
from humaidclf import plan_run_dirs, build_requests_jsonl_S

plan = plan_run_dirs(dataset_path, out_root="runs", model="gpt-4o-mini", tag="modeS-RULES5")
build_requests_jsonl_S(df, plan["requests_jsonl"], rules=RULES, model=plan["model"], temperature=0.0)
plan
```

This creates:

```
runs/<event>/<split>/gpt-4o-mini/<timestamp>-modeS-RULES5/
  requests.jsonl
  (later) outputs.jsonl, predictions.csv, analysis/
```

### E) Submit Batch, wait, download, parse

```python
import json

from humaidclf import (
    upload_file_for_batch,
    create_batch,
    wait_for_batch,
    download_file_content,
    parse_outputs_S_to_df,
    macro_f1,
)

# Submit batch and persist IDs so you can resume later
fid = upload_file_for_batch(str(plan["requests_jsonl"]))
bid = create_batch(fid, endpoint="/v1/chat/completions", completion_window="24h")

with open(plan["batch_meta_json"], "w", encoding="utf-8") as f:
    json.dump({"file_id": fid, "batch_id": bid}, f, indent=2)

# Poll until the batch is done (completed/failed/cancelled)
info = wait_for_batch(bid, poll_secs=20)

# Guard: only proceed if completed
status = info.get("status")
if status != "completed":
    raise RuntimeError(f"Batch ended with status='{status}'. Full info:\n{json.dumps(info, indent=2)}")

# Download outputs and parse to predictions.csv
out_file_id = info["output_file_id"]
download_file_content(out_file_id, str(plan["outputs_jsonl"]))

preds = parse_outputs_S_to_df(plan["outputs_jsonl"], df)
preds.to_csv(plan["predictions_csv"], index=False)
print("Saved predictions to:", plan["predictions_csv"])
print("Macro-F1:", macro_f1(preds))
```

Tip: to resume later without rebuilding, load plan["batch_meta_json"] and call wait_for_batch(bid) again; then download/parse as above.

```python
# Reload the saved metadata and poll again
with open(plan["batch_meta_json"], "r", encoding="utf-8") as f:
    meta = json.load(f)
bid = meta["batch_id"]

info = wait_for_batch(bid, poll_secs=20)
if info.get("status") != "completed":
    raise RuntimeError(f"Batch ended with status='{info.get('status')}'")

download_file_content(info["output_file_id"], str(plan["outputs_jsonl"]))
preds = parse_outputs_S_to_df(plan["outputs_jsonl"], df)
preds.to_csv(plan["predictions_csv"], index=False)
print("Saved predictions to:", plan["predictions_csv"])
print("Macro-F1:", macro_f1(preds))
```

---

## Analysis

### A) Analyze the current run

```python
from humaidclf import analyze_and_export_mistakes

mistakes_df, summary, per_cls, conf_df = analyze_and_export_mistakes(
    pred_csv_path=plan["predictions_csv"],
    out_mistakes_csv_path=plan["dir"] / "analysis" / "mistakes.csv",
    charts_dir=plan["dir"] / "analysis" / "charts",
)
summary
```

Artifacts appear under `runs/.../<run_id>/analysis/` as listed in the layout above.

### B) Analyze a past run from a path

```python
from pathlib import Path
from humaidclf import analyze_and_export_mistakes

base = Path("runs/california_wildfires_2018/train/gpt-4o-mini/20251017-220548-modeS-RULES5")
mistakes_df, summary, per_cls, conf_df = analyze_and_export_mistakes(
    pred_csv_path=base / "predictions.csv",
    out_mistakes_csv_path=base / "analysis" / "mistakes.csv",
    charts_dir=base / "analysis" / "charts",
)
summary
```

---

## Cost & Quality Tips

- Rules length vs cost: shorter rules reduce prompt tokens; keep only what moves the metric.
- `max_tokens`: 20–40 is enough for a single JSON label.
- Temperature: use 0.0 for deterministic classification.
- Macro-F1 vs Accuracy: Macro-F1 treats all classes equally; accuracy can be inflated by majority classes.
- Confusion matrices:
  - Counts: shows frequencies (can be dominated by big classes).
  - Row-normalized: shows per-class recall (diagonals = recall), fixed [0,1] with colorbar.

---

## Security & Git Tips

- Never commit `.env` or keys. If you did, rotate the key and remove it from history.
- Long commit messages: `git commit -m "title" --edit` (Vim: paste -> Esc :wq) or `git commit -F COMMIT_MESSAGE.txt`.

---

## License
Code is released under the MIT License (see `LICENSE`).

**Data:** This repository does not relicense any datasets. Use of HumAID or other datasets must comply with their original licenses/terms.  
**Services:** Use of the OpenAI API must comply with OpenAI's terms.

---

## Citation

If you use this code, please cite our forthcoming paper (details TBA).  
Provisional repository citation:

```bibtex
@misc{humaid-zeroshot-2025,
  title        = {Zero-shot HumAID Tweet Classification with OpenAI Batch},
  author       = {Anh Tran and Hongmin Li},
  year         = {2025},
  howpublished = {\url{https://github.com/anhtranst/humaid}},
  note         = {Version 0.1.0, MIT License}
}
```