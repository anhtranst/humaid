# humaidclf/budget.py
"""
Token budgeting utilities for OpenAI Batch runs.

Features:
- Estimate tokens per request (system + user[rules+tweet] + schema + output allowance)
- Estimate total tokens for a dataset (with sampling)
- Build a token index across datasets (decide which fit under a cap)
- Optional sharding: split a TSV into token-budgeted chunks

Notes:
- Uses tiktoken if available; otherwise falls back to a rough character-based heuristic.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Optional
import math
import pandas as pd

# These come from your existing package
from .io import load_tsv
from .prompts import SYSTEM_PROMPT
from .batch import SCHEMA_S

# Optional dependency
try:
    import tiktoken
except Exception:
    tiktoken = None


# ---------- Tokenizer helpers ----------

def get_token_encoder(model: str):
    """Return a tiktoken encoder for the model; fallback to cl100k_base, else None."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder=None) -> int:
    """Count tokens using tiktoken (if provided) or a simple char heuristic."""
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    # heuristic: ~4 chars/token is a common rough estimate for English
    return max(1, math.ceil(len(text) / 4))


# ---------- Estimation per request & per dataset ----------

def estimate_request_tokens(
    model: str,
    tweet_text: str,
    rules_text: str,
    max_output_tokens: int = 40,
) -> int:
    """
    Estimate tokens consumed per single request:
      system + user (rules + tweet) + response_format (schema) + output allowance.
    """
    enc = get_token_encoder(model)
    sys_tokens = count_tokens(SYSTEM_PROMPT, enc)

    user_msg = (
        "Allowed labels: (omitted here for brevity)\n"
        f"Rules:\n{rules_text}\n"
        "Choose exactly one label. If unrelated, choose 'not_humanitarian'.\n"
        f"Tweet: \"\"\"{tweet_text}\"\"\""
    )
    user_tokens = count_tokens(user_msg, enc)

    schema_tokens = count_tokens(str(SCHEMA_S), enc)

    return sys_tokens + user_tokens + schema_tokens + max_output_tokens


def estimate_dataset_tokens(
    tsv_path: str | Path,
    model: str,
    rules_text: str,
    sample_size: int = 200,
    max_output_tokens: int = 40,
) -> Dict[str, int]:
    """
    Estimate total token usage for a dataset by sampling up to `sample_size` rows.
    Returns: {"num_rows", "avg_req_tokens", "est_total_tokens"}
    """
    df = load_tsv(tsv_path)
    n = len(df)
    if n == 0:
        return {"num_rows": 0, "avg_req_tokens": 0, "est_total_tokens": 0}

    sample_n = min(sample_size, n)
    samp = df.sample(sample_n, random_state=42)["tweet_text"].tolist()

    per_req = [
        estimate_request_tokens(model, t, rules_text, max_output_tokens=max_output_tokens)
        for t in samp
    ]
    avg_tokens = int(sum(per_req) / len(per_req))
    total_est = int(avg_tokens * n)
    return {"num_rows": n, "avg_req_tokens": avg_tokens, "est_total_tokens": total_est}


def build_token_index(
    sources_df: pd.DataFrame,
    model: str,
    rules_text: str,
    batch_token_limit: int,
    safety_margin: float = 0.90,
    sample_size: int = 200,
    max_output_tokens: int = 40,
) -> pd.DataFrame:
    """
    Compute token estimates for each dataset entry in `sources_df`.
    Expects columns: event, split, tsv.
    Adds: num_rows, avg_req_tokens, est_total_tokens, fits_cap, limit_used_%
    """
    rows = []
    for _, r in sources_df.iterrows():
        stats = estimate_dataset_tokens(
            r["tsv"], model, rules_text, sample_size=sample_size, max_output_tokens=max_output_tokens
        )
        rows.append({
            "event": r["event"],
            "split": r["split"],
            "tsv": r["tsv"],
            **stats,
        })
    out = pd.DataFrame(rows)
    est_limit = int(batch_token_limit * safety_margin)
    out["fits_cap"] = out["est_total_tokens"] <= est_limit
    out["limit_used_%"] = (out["est_total_tokens"] / batch_token_limit * 100).round(1)
    return out.sort_values(["fits_cap", "est_total_tokens"], ascending=[False, True])


# ---------- Sharding utility (optional) ----------

def shard_dataset_by_tokens(
    tsv_path: str | Path,
    model: str,
    rules_text: str,
    target_token_budget: int,
    max_output_tokens: int = 40,
) -> list[pd.DataFrame]:
    """
    Split a TSV into a list of DataFrames where each shard's *estimated* token sum
    stays under target_token_budget (greedy, one pass).
    """
    df = load_tsv(tsv_path)
    if df.empty:
        return [df]

    enc = get_token_encoder(model)
    est_tokens = df["tweet_text"].apply(
        lambda t: estimate_request_tokens(model, t, rules_text, max_output_tokens=max_output_tokens)
    )

    shards = []
    current_rows = []
    current_budget = 0

    for idx, tok in est_tokens.items():
        if current_budget + tok > target_token_budget and current_rows:
            shards.append(df.loc[current_rows].copy())
            current_rows = []
            current_budget = 0
        current_rows.append(idx)
        current_budget += tok

    if current_rows:
        shards.append(df.loc[current_rows].copy())

    return shards
