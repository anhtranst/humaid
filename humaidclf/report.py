# humaidclf/report.py
"""
Reporting utilities for curated results with a top summary table and zoomable charts.

Directory layout you maintain manually:
  results/<event>/<split>/<model>/<run_name>/
    predictions.csv
    analysis/
      charts/
        confusion_matrix_counts.png
        confusion_matrix_row_normalized.png
        per_class_error_rate.png
        per_class_f1.png
        top_confusions.png   (optional)
        confusion_matrix.csv
        confusion_matrix_row_normalized.csv
        per_class_metrics.csv
        summary.json
      mistakes.csv

Then call build_results_index("results") to generate results/index.html.

Features:
- Summary table (Event, Split, Model, Run, Test size, Accuracy, Macro-F1)
- Detail cards with embedded chart previews
- Click any image to open a zoomable modal (ESC/backdrop click to close)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json
import shutil
import pandas as pd

# Images we try to embed (skipped if missing)
IMG_FILES = [
    "confusion_matrix_counts.png",
    "confusion_matrix_row_normalized.png",
    "per_class_error_rate.png",
    "per_class_f1.png",
    "top_confusions.png",  # optional
]

# ---------- Helpers for manual promotion (optional) ----------

def promote_run_to_results(run_dir: str | Path, results_root: str | Path, run_name: str | None = None) -> Path:
    """
    Copy one completed run into the curated results tree:

    Source (run_dir):
      runs/<event>/<split>/<model>/<timestamp-tag>/

    Destination (results_root):
      results/<event>/<split>/<model>/<run_name>/   # if run_name is None, use the source folder name

    Copies only 'predictions.csv' and 'analysis/' (keeps results lean).
    Returns the destination path.
    """
    run_dir = Path(run_dir)
    results_root = Path(results_root)

    model = run_dir.parent.name
    split = run_dir.parent.parent.name
    event = run_dir.parent.parent.parent.name
    run_id = run_dir.name
    run_name = run_name or run_id

    dest = results_root / event / split / model / run_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    src_pred = run_dir / "predictions.csv"
    src_analysis = run_dir / "analysis"
    if not src_pred.exists() or not src_analysis.exists():
        raise FileNotFoundError("Run directory must contain predictions.csv and analysis/")

    shutil.copy2(src_pred, dest / "predictions.csv")
    shutil.copytree(src_analysis, dest / "analysis", dirs_exist_ok=True)

    print("Promoted to:", dest)
    return dest

# ---------- Collector & index builder ----------

def _collect_results(results_root: Path) -> List[Dict]:
    """
    Traverse results tree and collect entries that have analysis/summary.json.

    Supported layouts:
    A) With run_name subfolder (preferred): results/<event>/<split>/<model>/<run_name>/
    B) Directly under model (fallback):    results/<event>/<split>/<model>/
    """
    entries: List[Dict] = []

    for event_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        event = event_dir.name
        for split_dir in sorted([p for p in event_dir.iterdir() if p.is_dir()]):
            split = split_dir.name
            for model_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                model = model_dir.name

                has_any = False
                # Case A: model/<run_name> subfolders
                for sub in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                    run_name = sub.name
                    analysis_dir = sub / "analysis"
                    summary_json = analysis_dir / "charts" / "summary.json"
                    if not summary_json.exists():
                        summary_json = analysis_dir / "summary.json"

                    if summary_json.exists():
                        has_any = True
                        with open(summary_json, "r", encoding="utf-8") as f:
                            summary = json.load(f)

                        entries.append({
                            "event": event,
                            "split": split,
                            "model": model,
                            "run_name": run_name,
                            "dir": str(sub),
                            "test_size": summary.get("num_total_with_truth", 0),
                            "accuracy": summary.get("accuracy", 0.0),
                            "macro_f1": summary.get("macro_f1", 0.0),
                            "charts_dir": str(analysis_dir / "charts") if (analysis_dir / "charts").exists() else str(analysis_dir),
                        })

                # Case B: directly under model (only if none found in subfolders)
                if not has_any:
                    analysis_dir = model_dir / "analysis"
                    summary_json = analysis_dir / "charts" / "summary.json"
                    if not summary_json.exists():
                        summary_json = analysis_dir / "summary.json"

                    if summary_json.exists():
                        with open(summary_json, "r", encoding="utf-8") as f:
                            summary = json.load(f)

                        entries.append({
                            "event": event,
                            "split": split,
                            "model": model,
                            "run_name": "",  # no subfolder
                            "dir": str(model_dir),
                            "test_size": summary.get("num_total_with_truth", 0),
                            "accuracy": summary.get("accuracy", 0.0),
                            "macro_f1": summary.get("macro_f1", 0.0),
                            "charts_dir": str(analysis_dir / "charts") if (analysis_dir / "charts").exists() else str(analysis_dir),
                        })

    return entries

def _render_summary_table(df: pd.DataFrame) -> str:
    """
    Render compact HTML table:
      Event | Split | Model | Run | Test size | Accuracy | Macro-F1
    """
    tbl = df.copy()
    tbl["Test size"] = tbl["test_size"].astype(int)
    tbl["Accuracy"] = tbl["accuracy"].map(lambda x: f"{x:.4f}")
    tbl["Macro-F1"] = tbl["macro_f1"].map(lambda x: f"{x:.4f}")

    tbl = tbl.rename(columns={
        "event": "Event",
        "split": "Split",
        "model": "Model",
        "run_name": "Run",
    })[["Event", "Split", "Model", "Run", "Test size", "Accuracy", "Macro-F1"]]

    rows = []
    for _, r in tbl.iterrows():
        rows.append(
            f"<tr>"
            f"<td><strong>{r['Event']}</strong></td>"
            f"<td>{r['Split']}</td>"
            f"<td><code>{r['Model']}</code></td>"
            f"<td><code>{r['Run']}</code></td>"
            f"<td class='num'>{r['Test size']}</td>"
            f"<td class='num'>{r['Accuracy']}</td>"
            f"<td class='num'>{r['Macro-F1']}</td>"
            f"</tr>"
        )
    return (
        "<table class='summary'>"
        "<thead><tr>"
        "<th>Event</th><th>Split</th><th>Model</th><th>Run</th>"
        "<th>Test size</th><th>Accuracy</th><th>Macro-F1</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

def build_results_index(results_root: str | Path, out_html: str | Path = None) -> pd.DataFrame:
    """
    Build an HTML index page summarizing curated results.
    - Top summary table
    - Detailed cards with embedded chart previews
    - Click any image to open a zoomable modal
    """
    results_root = Path(results_root)
    out_html = Path(out_html) if out_html else (results_root / "index.html")

    entries = _collect_results(results_root)
    if not entries:
        out_html.write_text("<h2>No results found.</h2>", encoding="utf-8")
        return pd.DataFrame()

    df = pd.DataFrame(entries).sort_values(["event", "split", "model", "run_name"]).reset_index(drop=True)

    def rel(p: Path | str) -> str:
        return str(Path(p).relative_to(results_root)).replace("\\", "/")

    # Summary
    summary_table_html = _render_summary_table(df)

    # Details (cards)
    rows_html: List[str] = []
    for _, r in df.iterrows():
        charts = Path(r["charts_dir"])
        imgs = []
        for name in IMG_FILES:
            fp = charts / name
            if fp.exists():
                imgs.append(
                    f'<div class="imgbox">'
                    f'  <img class="zoomable" src="{rel(fp)}" alt="{name}" '
                    f'       data-fullsrc="{rel(fp)}">'
                    f'</div>'
                )
        imgs_html = "\n".join(imgs) if imgs else "<em>No charts found.</em>"

        run_label = f" — <code>{r['run_name']}</code>" if r["run_name"] else ""
        rows_html.append(f"""
        <section class="card">
          <div class="head">
            <div>
              <h3>{r['event']} — {r['split']}</h3>
              <div class="sub">model: <code>{r['model']}</code>{run_label}</div>
              <div class="sub path">{rel(r['dir'])}</div>
            </div>
            <table class="metrics">
              <tr><th>Test size</th><td>{int(r['test_size'])}</td></tr>
              <tr><th>Accuracy</th><td>{r['accuracy']:.4f}</td></tr>
              <tr><th>Macro-F1</th><td>{r['macro_f1']:.4f}</td></tr>
            </table>
          </div>
          <div class="imgs">
            {imgs_html}
          </div>
        </section>
        """)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HumAID Zero-shot Results</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
  h1 {{ margin-top: 0; }}
  h2 {{ margin: 18px 0 8px; }}
  .summary {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; }}
  .summary th, .summary td {{ border: 1px solid #e5e7eb; padding: 8px 10px; }}
  .summary th {{ background: #f9fafb; text-align: left; }}
  .summary td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
  .head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
  .sub {{ color: #6b7280; font-size: 12px; }}
  .sub.path {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  table.metrics {{ border-collapse: collapse; }}
  table.metrics th {{ text-align: left; padding-right: 8px; color: #374151; }}
  table.metrics td {{ text-align: right; font-weight: 600; color: #111827; }}
  .imgs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-top: 12px; }}
  .imgbox {{ border: 1px solid #eee; border-radius: 8px; padding: 8px; background: #fafafa; }}
  .imgbox img {{ width: 100%; height: auto; display: block; cursor: zoom-in; }}

  /* Modal (lightbox) */
  .modal {{
    position: fixed; inset: 0; display: none;
    background: rgba(0,0,0,0.7); z-index: 9999;
    align-items: center; justify-content: center;
    padding: 24px;
  }}
  .modal.open {{ display: flex; }}
  .modal img {{
    max-width: 95vw; max-height: 95vh;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    border-radius: 8px; background: #fff;
  }}
  .modal .close {{
    position: absolute; top: 12px; right: 16px;
    font-size: 28px; color: #fff; cursor: pointer; user-select: none;
  }}
</style>
</head>
<body>
  <h1>HumAID Zero-shot Results</h1>

  <h2>Summary</h2>
  {summary_table_html}

  <h2>Details</h2>
  <div class="grid">
    {''.join(rows_html)}
  </div>

  <!-- Modal -->
  <div id="imgModal" class="modal" aria-hidden="true">
    <span class="close" title="Close (Esc)">&times;</span>
    <img id="modalImg" alt="Preview">
  </div>

  <script>
  // Simple lightbox for images with class="zoomable"
  (function() {{
    const modal = document.getElementById('imgModal');
    const modalImg = document.getElementById('modalImg');
    const closeBtn = modal.querySelector('.close');

    function openModal(src) {{
      modalImg.src = src;
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    }}

    function closeModal() {{
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
      modalImg.src = '';
    }}

    document.addEventListener('click', function(e) {{
      const img = e.target.closest('img.zoomable');
      if (img) {{
        const full = img.getAttribute('data-fullsrc') || img.src;
        openModal(full);
      }}
    }});

    // Close when clicking the backdrop or the close button
    modal.addEventListener('click', function(e) {{
      if (e.target === modal || e.target === closeBtn) {{
        closeModal();
      }}
    }});

    // ESC key
    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape' && modal.classList.contains('open')) {{
        closeModal();
      }}
    }});
  }})();
  </script>

  <footer style="margin-top:24px; color:#6b7280; font-size:12px;">
    Generated automatically. Click any chart to zoom. Press ESC to close.
  </footer>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    return df
