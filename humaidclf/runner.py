# humaidclf/runner.py
# -----------------------------------------------------------------------------
# Orchestrates a full zero-shot run for HumAID:
#   TSV -> (optional) dry-run sanity check -> build JSONL -> (maybe bypass) submit batch
#   -> (optional) wait -> download/parse -> PATCH MISSING/BLANK/OOS -> save predictions
#   -> (optional) analysis
#
# NEW:
#   • Single-label events bypass API exactly as before (local deterministic predictions).
#   • After Batch completes, we also fetch errors.jsonl (if any),
#     parse successes, mark explicit errors, and then PATCH any missing/blank/OOS
#     predictions synchronously so predictions.csv is row-aligned with the TSV.
# -----------------------------------------------------------------------------

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
    retry_fill_missing_predictions,
)
from .eval import macro_f1, analyze_and_export_mistakes


# =============================================================================
# Local helpers (runner-only)
# =============================================================================

def _present_labels_from_df(df: pd.DataFrame) -> list[str]:
    """
    Extract unique labels that appear in ground truth (cleaned).
    Sorting is only for determinism; downstream eval uses truth-only scope.
    """
    s = (
        df.get("class_label", pd.Series(dtype=object))
          .astype(str).str.strip()
          .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
          .dropna()
    )
    return sorted(set(s.tolist()))

def _predict_single_label_event(df: pd.DataFrame, only_label: str) -> pd.DataFrame:
    """
    Fast path for single-label events: no API call is needed.
    Produces the same columns as parse_outputs_S_to_df() (minus 'status').
    """
    return pd.DataFrame({
        "tweet_id": df["tweet_id"].astype(str),
        "tweet_text": df["tweet_text"],
        "class_label": df.get("class_label", ""),
        "predicted_label": only_label,
        "confidence": 1.0,       # deterministic since there is no choice
        "entropy": float("nan"),
        "status": "ok",
    })


# =============================================================================
# Public API
# =============================================================================

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
    End-to-end: load TSV -> dry-run -> build JSONL -> (maybe bypass) submit
    -> (optionally wait) -> download & parse -> PATCH -> save -> (optional) analysis.

    Notes
    -----
    - Blocks while waiting for batch completion unless submit_only=True.
    - Single-label events: build_requests_jsonl_S() writes an EMPTY file by convention;
      we detect that and predict locally (no API), then proceed with identical artifacts.
    """
    # -------------------------------------------------------------------------
    # 0) Load TSV
    # -------------------------------------------------------------------------
    df = load_tsv(dataset_path)

    # -------------------------------------------------------------------------
    # 1) Dry-run sanity check (small, synchronous)
    # -------------------------------------------------------------------------
    if dryrun_n and dryrun_n > 0:
        _ = sync_test_sample(
            df, n=dryrun_n, rules=rules, model=model,
            temperature=temperature, seed=42
        )

    # -------------------------------------------------------------------------
    # 2) Plan run dirs + build requests.jsonl
    # -------------------------------------------------------------------------
    plan = plan_run_dirs(dataset_path, out_root=out_root, model=model, tag=tag)

    # With filtered labels logic in batch.py, this writes an EMPTY file
    # if the event has exactly one valid label.
    build_requests_jsonl_S(
        df, plan["requests_jsonl"],
        rules=rules, model=model, temperature=temperature
    )

    # -------------------------------------------------------------------------
    # 2.1) Single-label BYPASS
    # -------------------------------------------------------------------------
    req_path = Path(plan["requests_jsonl"])
    if req_path.exists() and req_path.stat().st_size == 0:
        present = _present_labels_from_df(df)
        if len(present) != 1:
            raise RuntimeError(
                "build_requests_jsonl_S produced an empty JSONL but present label count != 1.\n"
                f"Detected labels (truth): {present}"
            )
        only_label = present[0]

        preds = _predict_single_label_event(df, only_label)

        Path(plan["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(plan["predictions_csv"], index=False)
        print("[single-label] Saved predictions to:", plan["predictions_csv"])
        try:
            print("[single-label] Macro-F1:", macro_f1(preds))
        except Exception:
            pass

        # Provenance for bypass mode
        meta = {"mode": "local_single_label", "only_label": only_label}
        with open(plan["batch_meta_json"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Optional analysis
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

    # -------------------------------------------------------------------------
    # 3) Submit batch (normal multi-label path)
    # -------------------------------------------------------------------------
    fid = upload_file_for_batch(str(plan["requests_jsonl"]))
    bid = create_batch(fid, endpoint="/v1/chat/completions", completion_window="24h")

    with open(plan["batch_meta_json"], "w", encoding="utf-8") as f:
        json.dump({"file_id": fid, "batch_id": bid}, f, indent=2)

    if submit_only:
        # Caller will resume later using resume_experiment()
        return plan, pd.DataFrame(), None

    # -------------------------------------------------------------------------
    # 4) Wait for completion, then download & parse
    #     • Successes -> outputs.jsonl
    #     • Errors    -> errors.jsonl  (if available)
    # -------------------------------------------------------------------------
    info = wait_for_batch(bid, poll_secs=poll_secs)
    status = info.get("status")
    if status != "completed":
        raise RuntimeError(
            f"Batch ended with status='{status}'. Full info:\n{json.dumps(info, indent=2)}"
        )

    out_file_id = info["output_file_id"]
    download_file_content(out_file_id, str(plan["outputs_jsonl"]))

    errors_jsonl_path: Optional[str] = None
    err_id = info.get("error_file_id")
    if err_id:
        errors_jsonl_path = str(Path(plan["dir"]) / "errors.jsonl")
        download_file_content(err_id, errors_jsonl_path)

    # Parse the provider's outputs and re-attach source fields
    preds = parse_outputs_S_to_df(
        plan["outputs_jsonl"], df,
        errors_jsonl_path=errors_jsonl_path
    )

    # -------------------------------------------------------------------------
    # 4.1) PATCH PASS: Fill any missing / blank / OOS predictions synchronously
    # -------------------------------------------------------------------------
    if (len(preds) != len(df)) or (preds["predicted_label"] == "").any() or (~preds["predicted_label"].isin(_present_labels_from_df(df))).any():
        preds = retry_fill_missing_predictions(
            source_df=df,
            preds_df=preds,
            rules=rules,
            model=model,
            temperature=temperature,
            max_tokens=40,
            max_retries=3,
            backoff_seconds=2.0,
        )

    # Persist final predictions (patched/aligned)
    Path(plan["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(plan["predictions_csv"], index=False)

    print("Saved predictions to:", plan["predictions_csv"])
    try:
        print("Macro-F1:", macro_f1(preds))  # eval default scope='truth'
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 5) Optional analysis artifacts
    # -------------------------------------------------------------------------
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
    Resume a previously submitted run by reading batch_meta.json from <run_dir>
    and finishing the download/parse/patch/analysis steps.

    Also supports single-label bypass runs where batch_meta.json contains:
      {"mode": "local_single_label", "only_label": "<label>"}.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "batch_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    plan = {
        "dir": run_dir,
        "requests_jsonl": run_dir / "requests.jsonl",
        "outputs_jsonl": run_dir / "outputs.jsonl",
        "predictions_csv": run_dir / "predictions.csv",
        "batch_meta_json": run_dir / "batch_meta.json",
    }

    # -------------------------------------------------------------------------
    # Single-label bypass resume
    # -------------------------------------------------------------------------
    if meta.get("mode") == "local_single_label":
        preds = pd.read_csv(plan["predictions_csv"])
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

    # -------------------------------------------------------------------------
    # Normal batch resume path
    # -------------------------------------------------------------------------
    bid = meta["batch_id"]

    info = wait_for_batch(bid, poll_secs=20)
    if info.get("status") != "completed":
        raise RuntimeError(f"Batch ended with status='{info.get('status')}'")

    # Re-download outputs (and errors if present) to ensure we have final files
    out_file_id = info["output_file_id"]
    download_file_content(out_file_id, str(plan["outputs_jsonl"]))

    errors_jsonl_path: Optional[str] = None
    err_id = info.get("error_file_id")
    if err_id:
        errors_jsonl_path = str(run_dir / "errors.jsonl")
        download_file_content(err_id, errors_jsonl_path)

    # Parse minimal + patch to row-align; we don't have the original df here,
    # so create a minimal shell to keep columns consistent.
    src_shell = pd.DataFrame(columns=["tweet_id", "tweet_text", "class_label"])
    preds = parse_outputs_S_to_df(
        str(plan["outputs_jsonl"]),
        src_shell,
        errors_jsonl_path=errors_jsonl_path
    )
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
