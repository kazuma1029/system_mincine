# classify_experiment.py
# -*- coding: utf-8 -*-
"""
experiment_all/{reviewer_id}/{movie_id}.xlsx のレビューを
allmodels・nounmodels・svmmodels で分類し、
accuracy / precision / recall / F-measure をレビュワーごとに xlsx 出力する。

ディレクトリ構成（想定）:
  experiment_all/{reviewer_id}/{movie_id}.xlsx  列0=レビュー文, 列1=評価値
  {MODELS_DIR}/{min_n}/allmodels/{reviewer_id}/
  {MODELS_DIR}/{min_n}/nounmodels/{reviewer_id}/{N}/
  {MODELS_DIR}/{min_n}/svmmodels/{reviewer_id}/

出力:
  results/results_allmodels_{min_n}.xlsx
  results/results_nounmodels_{min_n}.xlsx
  results/results_svmmodels_{min_n}.xlsx
"""

from collections import Counter
from math import log
from pathlib import Path

import joblib
import pandas as pd
import torch
from fugashi import Tagger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertJapaneseTokenizer

# ── パス設定（環境に合わせて変更） ────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
EXPERIMENT_DIR = BASE_DIR / "experiment_all"
MODELS_DIR     = Path(r"C:\Users\Oyabu\GoogleDriveStreaming\マイドライブ\models_sentences")
OUTPUT_DIR     = BASE_DIR / "results"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ── データ読み込み ────────────────────────────────────────────────────────────

def load_experiment(reviewer_id: int) -> tuple[list[str], list[float], list[str]]:
    """
    experiment_all/{reviewer_id}/ 内の全 xlsx を読み込む。
    戻り値: (texts, ratings, movie_ids)  ← 各要素は同インデックスで対応
    """
    reviewer_dir = EXPERIMENT_DIR / str(reviewer_id)
    texts, ratings, movie_ids = [], [], []

    for xlsx in sorted(reviewer_dir.glob("*.xlsx")):
        movie_id = xlsx.stem
        try:
            df = pd.read_excel(xlsx, header=None)
            for _, row in df.iterrows():
                text   = str(row.iloc[0]).strip()
                rating = pd.to_numeric(row.iloc[1], errors="coerce")
                if text and pd.notna(rating):
                    texts.append(text)
                    ratings.append(float(rating))
                    movie_ids.append(movie_id)
        except Exception as e:
            print(f"  [WARN] {xlsx.name}: {e}")

    return texts, ratings, movie_ids


def make_labels(ratings: list[float]) -> list[int]:
    """平均値以上を正例(1)、未満を負例(0)とする。"""
    avg = sum(ratings) / len(ratings)
    return [1 if r >= avg else 0 for r in ratings]


# ── 評価指標 ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": round(acc, 4), "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f1, 4)}


# ── BERT 推論 ─────────────────────────────────────────────────────────────────

def predict_bert(texts: list[str], model_dir: Path) -> list[int]:
    tokenizer = BertJapaneseTokenizer.from_pretrained(str(model_dir))
    model     = BertForSequenceClassification.from_pretrained(str(model_dir))
    model.to(DEVICE)
    model.eval()

    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc   = tokenizer(batch, truncation=True, padding=True,
                          max_length=512, return_tensors="pt")
        enc   = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().tolist())

    del model
    torch.cuda.empty_cache()
    return preds


# ── SVM 推論 ──────────────────────────────────────────────────────────────────

def _build_tfidf_from_experiment(reviewer_id: int) -> dict:
    """
    experiment_all/{reviewer_id}/{movie_id}.xlsx から映画単位のTF-IDFを計算する。
    学習時（_build_movie_tfidf）と同じ式5.1, 5.2 を使用。
    戻り値: {movie_id: {noun: tfidf_value}}
    """
    tagger = Tagger()
    movie_noun_counts: dict[str, Counter] = {}
    reviewer_dir = EXPERIMENT_DIR / str(reviewer_id)

    for xlsx in sorted(reviewer_dir.glob("*.xlsx")):
        movie_id = xlsx.stem
        try:
            df = pd.read_excel(xlsx, header=None)
            noun_count: Counter = Counter()
            for text in df.iloc[:, 0].astype(str):
                text = text.strip()
                if not text:
                    continue
                for w in tagger(text):
                    if "名詞" in w.feature:
                        noun_count[w.surface] += 1
            if noun_count:
                movie_noun_counts[movie_id] = noun_count
        except Exception as e:
            print(f"  [WARN] {xlsx.name}: {e}")

    N = len(movie_noun_counts)
    if N == 0:
        return {}

    df_count: Counter = Counter()
    for nc in movie_noun_counts.values():
        for noun in nc:
            df_count[noun] += 1

    idf = {noun: log(N / cnt, 2) + 1 for noun, cnt in df_count.items()}

    movie_tfidf: dict[str, dict] = {}
    for movie_id, nc in movie_noun_counts.items():
        total = sum(nc.values())
        movie_tfidf[movie_id] = {
            noun: (cnt / total) * idf[noun]
            for noun, cnt in nc.items()
        }
    return movie_tfidf


def predict_svm(texts: list[str], movie_ids: list[str],
                reviewer_id: int, model_dir: Path) -> list[int]:
    """
    映画単位TF-IDFで特徴ベクトルを構築し、保存済み LinearSVC で予測する。
    """
    clf = joblib.load(model_dir / "svm_model.pkl")
    dv  = joblib.load(model_dir / "dict_vectorizer.pkl")
    tagger = Tagger()

    movie_tfidf = _build_tfidf_from_experiment(reviewer_id)

    vectors = []
    for text, movie_id in zip(texts, movie_ids):
        tfidf = movie_tfidf.get(movie_id, {})
        vec = {
            w.surface: tfidf[w.surface]
            for w in tagger(str(text).strip())
            if "名詞" in w.feature and w.surface in tfidf
        }
        vectors.append(vec if vec else {"__empty__": 0.0})

    X = dv.transform(vectors)
    return clf.predict(X).tolist()


# ── モデル探索 ────────────────────────────────────────────────────────────────

def find_min_movie_counts() -> list[int]:
    return sorted([
        int(p.name) for p in MODELS_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ])


def reviewer_ids_in_experiment() -> list[int]:
    return sorted([
        int(p.name) for p in EXPERIMENT_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ])


# ── 評価ループ ────────────────────────────────────────────────────────────────

def evaluate_allmodels(min_n: int, reviewer_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows  = []
    model_base   = MODELS_DIR / str(min_n) / "allmodels"
    if not model_base.exists():
        return pd.DataFrame(), pd.DataFrame()

    for rid in reviewer_ids:
        model_dir = model_base / str(rid)
        if not model_dir.exists():
            continue
        exp_dir = EXPERIMENT_DIR / str(rid)
        if not exp_dir.exists():
            continue
        try:
            texts, ratings, movie_ids = load_experiment(rid)
            if not texts:
                continue
            labels = make_labels(ratings)
            preds  = predict_bert(texts, model_dir)
            m      = compute_metrics(labels, preds)
            summary_rows.append({"reviewer_id": rid, **m})
            for text, movie_id, rating, true_label, pred_label in zip(
                    texts, movie_ids, ratings, labels, preds):
                detail_rows.append({
                    "reviewer_id": rid,
                    "movie_id":    movie_id,
                    "review":      text,
                    "rating":      rating,
                    "true_label":  true_label,
                    "pred_label":  pred_label,
                })
            print(f"  reviewer {rid}: {m}")
        except Exception as e:
            print(f"  [WARN] reviewer {rid}: {e}")

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def evaluate_nounmodels(min_n: int, reviewer_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows  = []
    model_base   = MODELS_DIR / str(min_n) / "nounmodels"
    if not model_base.exists():
        return pd.DataFrame(), pd.DataFrame()

    for rid in reviewer_ids:
        reviewer_model_dir = model_base / str(rid)
        if not reviewer_model_dir.exists():
            continue
        exp_dir = EXPERIMENT_DIR / str(rid)
        if not exp_dir.exists():
            continue
        try:
            texts, ratings, movie_ids = load_experiment(rid)
            if not texts:
                continue
            labels = make_labels(ratings)
        except Exception as e:
            print(f"  [WARN] reviewer {rid} データ読み込み失敗: {e}")
            continue

        for n_dir in sorted(reviewer_model_dir.iterdir(),
                            key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if not n_dir.name.isdigit():
                continue
            top_n = int(n_dir.name)
            try:
                preds = predict_bert(texts, n_dir)
                m     = compute_metrics(labels, preds)
                summary_rows.append({"reviewer_id": rid, "top_n": top_n, **m})
                for text, movie_id, rating, true_label, pred_label in zip(
                        texts, movie_ids, ratings, labels, preds):
                    detail_rows.append({
                        "reviewer_id": rid,
                        "top_n":       top_n,
                        "movie_id":    movie_id,
                        "review":      text,
                        "rating":      rating,
                        "true_label":  true_label,
                        "pred_label":  pred_label,
                    })
                print(f"  reviewer {rid} N={top_n}: {m}")
            except Exception as e:
                print(f"  [WARN] reviewer {rid} N={top_n}: {e}")

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def evaluate_svmmodels(min_n: int, reviewer_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows  = []
    model_base   = MODELS_DIR / str(min_n) / "svmmodels"
    if not model_base.exists():
        return pd.DataFrame(), pd.DataFrame()

    for rid in reviewer_ids:
        model_dir = model_base / str(rid)
        if not model_dir.exists():
            continue
        exp_dir = EXPERIMENT_DIR / str(rid)
        if not exp_dir.exists():
            continue
        try:
            texts, ratings, movie_ids = load_experiment(rid)
            if not texts:
                continue
            labels = make_labels(ratings)
            preds  = predict_svm(texts, movie_ids, rid, model_dir)
            m      = compute_metrics(labels, preds)
            summary_rows.append({"reviewer_id": rid, **m})
            for text, movie_id, rating, true_label, pred_label in zip(
                    texts, movie_ids, ratings, labels, preds):
                detail_rows.append({
                    "reviewer_id": rid,
                    "movie_id":    movie_id,
                    "review":      text,
                    "rating":      rating,
                    "true_label":  true_label,
                    "pred_label":  pred_label,
                })
            print(f"  reviewer {rid}: {m}")
        except Exception as e:
            print(f"  [WARN] reviewer {rid}: {e}")

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    min_counts   = find_min_movie_counts()
    reviewer_ids = reviewer_ids_in_experiment()

    if not min_counts:
        print("[ERROR] modelsディレクトリ内にmin_movie_countディレクトリが見つかりません。")
        return
    print(f"[INFO] min_movie_count: {min_counts}")
    print(f"[INFO] 対象レビュワー数: {len(reviewer_ids)}")

    for min_n in min_counts:
        print(f"\n{'='*60}")
        print(f"[min_movie_count = {min_n}]")

        # print("\n[svmmodels]")
        # df_summary, df_detail = evaluate_svmmodels(min_n, reviewer_ids)
        # if not df_summary.empty:
        #     df_summary.to_excel(OUTPUT_DIR / f"results_svmmodels_{min_n}.xlsx", index=False)
        #     print(f"  → 保存: results_svmmodels_{min_n}.xlsx")
        # if not df_detail.empty:
        #     df_detail.to_csv(OUTPUT_DIR / f"detail_svmmodels_{min_n}.csv", index=False, encoding="utf-8-sig")
        #     print(f"  → 保存: detail_svmmodels_{min_n}.csv")

        print("\n[nounmodels]")
        df_summary, df_detail = evaluate_nounmodels(min_n, reviewer_ids)
        if not df_summary.empty:
            df_summary.to_excel(OUTPUT_DIR / f"results_nounmodels_{min_n}.xlsx", index=False)
            print(f"  → 保存: results_nounmodels_{min_n}.xlsx")
        if not df_detail.empty:
            df_detail.to_csv(OUTPUT_DIR / f"detail_nounmodels_{min_n}.csv", index=False, encoding="utf-8-sig")
            print(f"  → 保存: detail_nounmodels_{min_n}.csv")

        print("\n[allmodels]")
        df_summary, df_detail = evaluate_allmodels(min_n, reviewer_ids)
        if not df_summary.empty:
            df_summary.to_excel(OUTPUT_DIR / f"results_allmodels_{min_n}.xlsx", index=False)
            print(f"  → 保存: results_allmodels_{min_n}.xlsx")
        if not df_detail.empty:
            df_detail.to_csv(OUTPUT_DIR / f"detail_allmodels_{min_n}.csv", index=False, encoding="utf-8-sig")
            print(f"  → 保存: detail_allmodels_{min_n}.csv")

    print("\n[DONE]")


if __name__ == "__main__":
    main()