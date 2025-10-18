import os, json, time, requests
import pandas as pd

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in your environment (e.g., via dotenv)."

OPENAI_BASE = "https://api.openai.com/v1"
H_JSON = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
H_MULTI = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

from .prompts import SYSTEM_PROMPT, make_user_message, LABELS

# Structured Outputs schema (Mode S)
SCHEMA_S = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["label"],
    "additionalProperties": False
}

def sync_test_sample(
    df: pd.DataFrame,
    n: int = 5,
    rules: str = "",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    seed: int = 1,
) -> pd.DataFrame:
    """Zero-shot sanity check on a small sample."""
    from .eval import macro_f1
    test = df.sample(min(n, len(df)), random_state=seed).copy()
    rows = []
    for _, r in test.iterrows():
        user_msg = make_user_message(str(r["tweet_text"]), rules, LABELS)
        body = {
            "model": model,
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": 40,  # chat.completions uses max_tokens
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "tweet_label", "schema": SCHEMA_S}},
        }
        resp = requests.post(f"{OPENAI_BASE}/chat/completions", headers=H_JSON, json=body, timeout=60)
        if resp.status_code != 200:
            print(">>> API error body:", resp.text)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        parsed = choice.get("parsed")
        if not parsed:
            content = choice.get("content", "")
            parsed = json.loads(content) if content else {}
        rows.append({
            "tweet_id": r["tweet_id"],
            "tweet_text": r["tweet_text"],
            "class_label": r.get("class_label", ""),
            "predicted_label": parsed.get("label", ""),
            "confidence": parsed.get("confidence", None),
            "entropy": float("nan"),
        })
    out = pd.DataFrame(rows)
    print("Macro-F1 (tiny sample):", macro_f1(out))
    return out

def build_requests_jsonl_S(
    df: pd.DataFrame,
    out_path: str,
    rules: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
):
    """Zero-shot Batch JSONL builder targeting /v1/chat/completions."""
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            tid = str(row["tweet_id"]).strip()
            text = str(row["tweet_text"] or "").replace("\r", " ").strip()
            user_msg = make_user_message(text, rules, LABELS)
            body = {
                "model": model,
                "temperature": temperature,
                "top_p": 1,
                "max_tokens": 40,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "response_format": {"type": "json_schema", "json_schema": {"name": "tweet_label", "schema": SCHEMA_S}},
            }
            line = {
                "custom_id": f"tweet-{tid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return out_path

# Batch helpers
def upload_file_for_batch(filepath: str) -> str:
    with open(filepath, "rb") as f:
        r = requests.post(f"{OPENAI_BASE}/files", headers=H_MULTI,
                          files={"file": (os.path.basename(filepath), f)}, data={"purpose": "batch"}, timeout=60)
    r.raise_for_status()
    return r.json()["id"]

def create_batch(input_file_id: str, endpoint="/v1/chat/completions", completion_window="24h") -> str:
    payload = {"input_file_id": input_file_id, "endpoint": endpoint, "completion_window": completion_window}
    r = requests.post(f"{OPENAI_BASE}/batches", headers=H_JSON, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["id"]

def get_batch(batch_id: str) -> dict:
    r = requests.get(f"{OPENAI_BASE}/batches/{batch_id}", headers=H_JSON, timeout=60)
    r.raise_for_status()
    return r.json()

def wait_for_batch(batch_id: str, poll_secs=20) -> dict:
    while True:
        info = get_batch(batch_id)
        status = info.get("status")
        print(f"[batch {batch_id}] status = {status}")
        if status in {"completed", "failed", "cancelled"}:
            return info
        time.sleep(poll_secs)

def download_file_content(file_id: str, out_path: str) -> str:
    r = requests.get(f"{OPENAI_BASE}/files/{file_id}/content", headers=H_JSON, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def parse_outputs_S_to_df(outputs_jsonl_path: str, source_df: pd.DataFrame) -> pd.DataFrame:
    by_id = {
        str(r["tweet_id"]): {"tweet_text": r["tweet_text"], "class_label": r.get("class_label", "")}
        for _, r in source_df.iterrows()
    }
    rows = []
    with open(outputs_jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid = rec.get("custom_id", "").replace("tweet-", "")
            choice = rec["response"]["body"]["choices"][0]["message"]
            parsed = choice.get("parsed")
            if not parsed:
                content = choice.get("content", "")
                if isinstance(content, list):
                    content = content[0].get("text", "")
                parsed = json.loads(content) if content else {}
            local = by_id.get(tid, {"tweet_text": "", "class_label": ""})
            rows.append({
                "tweet_id": tid,
                "tweet_text": local["tweet_text"],
                "class_label": local["class_label"],
                "predicted_label": parsed.get("label", ""),
                "confidence": parsed.get("confidence", None),
                "entropy": float("nan"),
            })
    return pd.DataFrame(rows, columns=["tweet_id", "tweet_text", "class_label", "predicted_label", "confidence", "entropy"])
