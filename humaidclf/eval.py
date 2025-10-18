import pathlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def macro_f1(df, truth_col="class_label", pred_col="predicted_label"):
    sub = df[(df[truth_col] != "") & (df[pred_col] != "")]
    if sub.empty: return float("nan")
    labels = sorted(set(sub[truth_col]) | set(sub[pred_col]))
    f1s = []
    for c in labels:
        tp = ((sub[truth_col] == c) & (sub[pred_col] == c)).sum()
        fp = ((sub[truth_col] != c) & (sub[pred_col] == c)).sum()
        fn = ((sub[truth_col] == c) & (sub[pred_col] != c)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
        f1s.append(f1)
    return sum(f1s)/len(f1s)

def analyze_and_export_mistakes(
    pred_csv_path: str,
    out_mistakes_csv_path: str,
    charts_dir: str | None = None,
    truth_col: str = "class_label",
    pred_col: str = "predicted_label",
    id_col: str = "tweet_id",
    text_col: str = "tweet_text",
    save_summary_json: bool = True,
):
    df = pd.read_csv(pred_csv_path)
    for c in (truth_col, pred_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {pred_csv_path}")

    df_eval = df[df[truth_col].astype(str).str.len() > 0].copy()
    mistakes_df = df_eval.loc[df_eval[truth_col] != df_eval[pred_col]].copy()

    out_p = pathlib.Path(out_mistakes_csv_path); out_p.parent.mkdir(parents=True, exist_ok=True)
    mistakes_df.to_csv(out_p, index=False)

    labels = sorted(set(df_eval[truth_col]) | set(df_eval[pred_col]))
    conf_mat_df = pd.crosstab(df_eval[truth_col], df_eval[pred_col], dropna=False)\
                    .reindex(index=labels, columns=labels, fill_value=0)

    tp = np.diag(conf_mat_df.values)
    support_true = conf_mat_df.sum(axis=1).values
    support_pred = conf_mat_df.sum(axis=0).values
    fp = support_pred - tp
    fn = support_true - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1        = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp, dtype=float), where=(precision+recall) != 0)
    error_rate = np.divide(fn + fp, support_true + fp, out=np.zeros_like(tp, dtype=float), where=(support_true + fp) != 0)

    per_class_df = pd.DataFrame({
        "label": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support_true.astype(int),
        "error_rate": error_rate
    }).sort_values("label")

    accuracy = (tp.sum() / conf_mat_df.values.sum()) if conf_mat_df.values.sum() else 0.0
    macro = float(per_class_df["f1"].mean()) if not per_class_df.empty else 0.0

    summary = {
        "num_total_with_truth": int(len(df_eval)),
        "num_correct": int(tp.sum()),
        "num_incorrect": int(len(mistakes_df)),
        "accuracy": accuracy,
        "macro_f1": macro,
        "labels": labels,
    }

    if charts_dir:
        charts_dir = pathlib.Path(charts_dir); charts_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix (counts)
        plt.figure(figsize=(8 + 0.3*len(labels), 6 + 0.3*len(labels)))
        plt.imshow(conf_mat_df.values, interpolation="nearest")
        plt.title("Confusion Matrix (counts)")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.tight_layout(); plt.savefig(charts_dir / "confusion_matrix.png", dpi=200); plt.close()

        # Per-class F1
        plt.figure(figsize=(max(8, 0.6*len(labels)), 5))
        plt.bar(per_class_df["label"], per_class_df["f1"])
        plt.title("Per-class F1"); plt.xticks(rotation=90); plt.ylim(0,1)
        plt.tight_layout(); plt.savefig(charts_dir / "per_class_f1.png", dpi=200); plt.close()

        # Error rate
        plt.figure(figsize=(max(8, 0.6*len(labels)), 5))
        plt.bar(per_class_df["label"], per_class_df["error_rate"])
        plt.title("Per-class Error Rate"); plt.xticks(rotation=90); plt.ylim(0,1)
        plt.tight_layout(); plt.savefig(charts_dir / "per_class_error_rate.png", dpi=200); plt.close()

        # Top confusions (off-diagonal)
        C = conf_mat_df.values
        pairs = [(labels[i], labels[j], int(C[i,j])) for i in range(len(labels)) for j in range(len(labels)) if i!=j and C[i,j]>0]
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_k = pairs[:15]
        if top_k:
            plt.figure(figsize=(10, max(4, 0.4*len(top_k))))
            ylabels = [f"{t} → {p}" for t,p,_ in top_k]
            counts = [c for _,_,c in top_k]
            y = range(len(top_k))
            plt.barh(list(y), counts); plt.yticks(list(y), ylabels); plt.gca().invert_yaxis()
            plt.title("Top Confusions (off-diagonal)"); plt.tight_layout()
            plt.savefig(charts_dir / "top_confusions.png", dpi=200); plt.close()

        per_class_df.to_csv(charts_dir / "per_class_metrics.csv", index=False)
        conf_mat_df.to_csv(charts_dir / "confusion_matrix.csv")
        if save_summary_json:
            with open(charts_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    return mistakes_df, summary, per_class_df, conf_mat_df
