# humaidclf/runner_sharded.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import json

import pandas as pd

from .io import load_tsv, plan_run_dirs
from .stratify import stratified_k_shards
from .runner import _present_labels_from_df  # reuse helper
from .batch import (
    build_requests_jsonl_S,
    upload_file_for_batch, create_batch, wait_for_batch,
    download_file_content, parse_outputs_S_to_df,
    retry_fill_missing_predictions,
)
from .eval import analyze_and_export_mistakes, macro_f1


def run_experiment_sharded(
    dataset_path: str,
    rules: str,
    model: str = "gpt-4o-mini",
    tag: str = "modeS-sharded",
    *,
    k_shards: int = 4,
    temperature: float = 0.0,
    poll_secs: int = 60,
    out_root: str = "runs",
    do_analysis: bool = True,
    analysis_subdir: str = "analysis",
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Stratify the dataset into k shards (preserving class ratios), run each shard end-to-end,
    then merge predictions. IMPORTANT:
      • All shards share ONE consistent, event-level label set (labels_override).
      • Single-label bypass is DISABLED for shards to avoid accidental “easy-mode”.

    Returns
    -------
    plan_root : dict of run paths
    merged    : merged predictions DataFrame (row-aligned to original TSV order)
    summary   : optional analysis summary dict (if do_analysis=True), else None
    """
    # --- Load full dataset and compute event-level labels
    df_full = load_tsv(dataset_path).reset_index(drop=False).rename(columns={"index": "_order"})
    # Ensure tweet_id is string for stable merges
    df_full["tweet_id"] = df_full["tweet_id"].astype(str)

    truth_labels = _present_labels_from_df(df_full)  # event-level label set (truth-only)

    # --- Plan directory for the whole sharded run
    plan_root = plan_run_dirs(dataset_path, out_root=out_root, model=model, tag=tag)
    run_dir = Path(plan_root["dir"])
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)

    # --- Build stratified shards
    shards = stratified_k_shards(df_full, label_col="class_label", k=k_shards, seed=42)

    all_preds: List[pd.DataFrame] = []
    for i, df_shard in enumerate(shards, start=1):
        shard_name = f"shard{i:02d}"
        shard_dir = run_dir / "shards" / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Write temporary TSV for this shard (for provenance/debugging)
        shard_tsv = shard_dir / f"{shard_name}.tsv"
        df_shard.to_csv(shard_tsv, sep="\t", index=False)

        # Mini plan for this shard
        plan = {
            "dir": shard_dir,
            "requests_jsonl": shard_dir / "requests.jsonl",
            "outputs_jsonl":  shard_dir / "outputs.jsonl",
            "predictions_csv": shard_dir / "predictions.csv",
            "batch_meta_json": shard_dir / "batch_meta.json",
        }

        # --- Build requests with event-level labels; DISABLE single-label bypass for shards
        build_requests_jsonl_S(
            df_shard,
            out_path=str(plan["requests_jsonl"]),
            rules=rules,
            model=model,
            temperature=temperature,
            labels_override=truth_labels,          # keep schema consistent across shards
            allow_single_label_bypass=False,       # never bypass in sharded mode
        )

        # --- Submit + wait
        fid = upload_file_for_batch(str(plan["requests_jsonl"]))
        bid = create_batch(fid, endpoint="/v1/chat/completions", completion_window="24h")
        (shard_dir / "batch_meta.json").write_text(
            json.dumps({"file_id": fid, "batch_id": bid}, indent=2), encoding="utf-8"
        )

        info = wait_for_batch(bid, poll_secs=poll_secs)
        
        # Always persist batch info for post-mortem
        (shard_dir / "batch_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        
        err_path = None
        if info.get("error_file_id"):
            err_path = shard_dir / "errors.jsonl"
            download_file_content(info["error_file_id"], str(err_path))
        
        if info.get("status") != "completed":
            raise RuntimeError(
                f"[{shard_name}] batch ended with status='{info.get('status')}'. "
                f"See '{shard_dir / 'batch_info.json'}' and "
                f"{(err_path and str(err_path)) or 'Batch dashboard'} for details."
            )

        download_file_content(info["output_file_id"], str(plan["outputs_jsonl"]))

        # --- Parse + patch missing
        # Ensure shard tweet_id dtype is string to preserve alignment
        df_shard = df_shard.copy()
        df_shard["tweet_id"] = df_shard["tweet_id"].astype(str)
        
        errors_jsonl_path = str(err_path) if err_path else None
        
        preds = parse_outputs_S_to_df(
            outputs_jsonl_path=str(plan["outputs_jsonl"]),
            source_df=df_shard,
            errors_jsonl_path=errors_jsonl_path
        )

        if len(preds) != len(df_shard) or (preds["predicted_label"] == "").any():
            preds = retry_fill_missing_predictions(
                source_df=df_shard,
                preds_df=preds,
                rules=rules,
                model=model,
                temperature=temperature,
                max_tokens=40,
                max_retries=3,
                backoff_seconds=2.0,
                labels_override=truth_labels,   # keep the same label space for retries
            )

        preds.to_csv(plan["predictions_csv"], index=False)
        all_preds.append(preds)

    # --- Merge shard predictions and restore original TSV order
    merged = pd.concat(all_preds, ignore_index=True)
    merged["tweet_id"] = merged["tweet_id"].astype(str)

    # Reattach original order key and sort by it
    order_map = df_full[["tweet_id", "_order"]].copy()
    merged = merged.merge(order_map, on="tweet_id", how="left").sort_values("_order").drop(columns=["_order"])

    # --- Save final merged artifacts at the root run dir
    merged.to_csv(plan_root["predictions_csv"], index=False)

    # --- Optional analysis on merged predictions
    analysis_summary = None
    if do_analysis:
        charts_dir = run_dir / analysis_subdir / "charts"
        mistakes_csv = run_dir / analysis_subdir / "mistakes.csv"
        _, summary, _, _ = analyze_and_export_mistakes(
            pred_csv_path=str(plan_root["predictions_csv"]),
            out_mistakes_csv_path=str(mistakes_csv),
            charts_dir=str(charts_dir),
            # scope='truth' by default in eval.py
        )
        analysis_summary = summary

    print("Saved merged predictions to:", plan_root["predictions_csv"])
    print("Macro-F1 (merged):", macro_f1(merged))

    return plan_root, merged, analysis_summary
