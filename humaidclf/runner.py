# humaidclf/runner.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd

from .io import load_tsv, plan_run_dirs
from .batch import (
    sync_test_sample,
    build_requests_jsonl_S,
    upload_file_for_batch,
    create_batch,
    wait_for_batch,
    download_file_content,
    parse_outputs_S_to_df,
)
from .eval import macro_f1, analyze_and_export_mistakes


def run_experiment(
    dataset_path: str,
    rules: str,
    model: str = "gpt-4o-mini",
    tag: str = "modeS",
    *,
    temperature: float = 0.0,
    dryrun_n: int = 20,
    poll_secs: int = 60,
    out_root: str = "runs",
    do_analysis: bool = True,
    analysis_subdir: str = "analysis",
    submit_only: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    End-to-end: load TSV -> dry-run sanity check -> build batch JSONL -> submit -> (optionally wait) ->
    download + parse -> save predictions -> (optionally) analysis.

    Parameters
    ----------
    dataset_path : str
        Path to input TSV with columns: tweet_id, tweet_text, (optional) class_label.
    rules : str
        Zero-shot rules text used in the user message.
    model : str
        OpenAI model name (e.g., "gpt-4o-mini").
    tag : str
        Tag appended to the run directory name (timestamp is auto-added).
    temperature : float
        Generation temperature (use 0.0 for deterministic classification).
    dryrun_n : int
        Number of examples for the synchronous sanity check before submitting the batch.
    poll_secs : int
        Seconds between batch status polls (if submit_only=False).
    out_root : str
        Root folder for run outputs (the function creates "runs/<event>/<split>/<model>/<timestamp>-<tag>/").
    do_analysis : bool
        If True, writes analysis artifacts under <run_dir>/<analysis_subdir>/.
    analysis_subdir : str
        Subfolder name for analysis inside the run directory (default: "analysis").
    submit_only : bool
        If True, submit batch and return immediately after writing batch_meta.json (no waiting, no parsing).

    Returns
    -------
    plan : dict
        Paths & metadata for the run (includes 'dir', 'requests_jsonl', 'outputs_jsonl', 'predictions_csv', 'batch_meta_json').
    preds_df : pandas.DataFrame
        Predictions dataframe (empty if submit_only=True).
    analysis_summary : dict | None
        Summary dict from analyze_and_export_mistakes (None if do_analysis=False or submit_only=True).

    Notes
    -----
    - This function blocks while waiting for batch completion unless submit_only=True.
    - Use the saved batch_meta.json to resume later with wait_for_batch(bid).
    """
    # 0) Load TSV
    df = load_tsv(dataset_path)

    # 1) Dry-run sanity check (small sample to catch schema/prompt issues early)
    if dryrun_n and dryrun_n > 0:
        _ = sync_test_sample(df, n=dryrun_n, rules=rules, model=model, temperature=temperature, seed=42)

    # 2) Plan run dirs + build requests.jsonl
    plan = plan_run_dirs(dataset_path, out_root=out_root, model=model, tag=tag)
    build_requests_jsonl_S(df, plan["requests_jsonl"], rules=rules, model=model, temperature=temperature)

    # 3) Submit batch
    fid = upload_file_for_batch(str(plan["requests_jsonl"]))
    bid = create_batch(fid, endpoint="/v1/chat/completions", completion_window="24h")
    with open(plan["batch_meta_json"], "w", encoding="utf-8") as f:
        json.dump({"file_id": fid, "batch_id": bid}, f, indent=2)

    if submit_only:
        # Return early so caller can resume later
        return plan, pd.DataFrame(), None

    # Wait for completion
    info = wait_for_batch(bid, poll_secs=poll_secs)
    status = info.get("status")
    if status != "completed":
        raise RuntimeError(f"Batch ended with status='{status}'. Full info:\n{json.dumps(info, indent=2)}")

    # 4) Download + parse + save
    out_file_id = info["output_file_id"]
    download_file_content(out_file_id, str(plan["outputs_jsonl"]))
    preds = parse_outputs_S_to_df(plan["outputs_jsonl"], df)
    Path(plan["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(plan["predictions_csv"], index=False)

    print("Saved predictions to:", plan["predictions_csv"])
    print("Macro-F1:", macro_f1(preds))

    # Optional: Analysis artifacts into <run_dir>/<analysis_subdir>/
    analysis_summary = None
    if do_analysis:
        charts_dir = Path(plan["dir"]) / analysis_subdir / "charts"
        mistakes_csv = Path(plan["dir"]) / analysis_subdir / "mistakes.csv"
        _, summary, _, _ = analyze_and_export_mistakes(
            pred_csv_path=str(plan["predictions_csv"]),
            out_mistakes_csv_path=str(mistakes_csv),
            charts_dir=str(charts_dir),
        )
        analysis_summary = summary

    return plan, preds, analysis_summary


def resume_experiment(
    run_dir: str | Path,
    *,
    do_analysis: bool = True,
    analysis_subdir: str = "analysis",
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Resume a previously submitted run by reading batch_meta.json from <run_dir> and finishing
    the download/parse/analysis steps (assumes the batch is now completed).

    Parameters
    ----------
    run_dir : str | Path
        The existing run directory created by plan_run_dirs (contains batch_meta.json).
    do_analysis : bool
        Whether to produce analysis artifacts under <run_dir>/<analysis_subdir>/.
    analysis_subdir : str
        Subfolder for analysis output.

    Returns
    -------
    plan : dict
        Same structure as returned by run_experiment().
    preds_df : pandas.DataFrame
        Parsed predictions.
    analysis_summary : dict | None
        Summary dict from analyze_and_export_mistakes (None if do_analysis=False).
    """
    run_dir = Path(run_dir)
    with open(run_dir / "batch_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    bid = meta["batch_id"]

    # Rebuild "plan" mapping
    plan = {
        "dir": run_dir,
        "requests_jsonl": run_dir / "requests.jsonl",
        "outputs_jsonl": run_dir / "outputs.jsonl",
        "predictions_csv": run_dir / "predictions.csv",
        "batch_meta_json": run_dir / "batch_meta.json",
    }

    # Poll again to get final info, then download/parse
    info = wait_for_batch(bid, poll_secs=20)
    if info.get("status") != "completed":
        raise RuntimeError(f"Batch ended with status='{info.get('status')}'")

    out_file_id = info["output_file_id"]
    download_file_content(out_file_id, str(plan["outputs_jsonl"]))

    # We need the original dataset rows to parse text/labels back in; for resume we assume
    # you have the original TSV path recorded externally or you only need the parsed fields.
    # If you want full reattachment of tweet_text/class_label, consider saving a compact
    # snapshot of the source rows at plan time.
    # For now, parse with minimal fields and let downstream code join if needed.
    # (If your parse requires the original df, extend batch_meta.json to include dataset_path.)
    preds = parse_outputs_S_to_df(plan["outputs_jsonl"], pd.DataFrame(columns=["tweet_id","tweet_text","class_label"]))
    preds.to_csv(plan["predictions_csv"], index=False)

    analysis_summary = None
    if do_analysis:
        charts_dir = run_dir / analysis_subdir / "charts"
        mistakes_csv = run_dir / analysis_subdir / "mistakes.csv"
        _, summary, _, _ = analyze_and_export_mistakes(
            pred_csv_path=str(plan["predictions_csv"]),
            out_mistakes_csv_path=str(mistakes_csv),
            charts_dir=str(charts_dir),
        )
        analysis_summary = summary

    return plan, preds, analysis_summary
