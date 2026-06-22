# finetune_all_reviewers_126.py
# -*- coding: utf-8 -*-
"""
レビュワー 126~250 に対して reviewer_summary.xlsx から好み映画・好みでない映画を参照し、
reviews_posinega/ の polarity==1 文を使って BERT をファインチューニングする。

データ構成:
  正例 (label=1): 好み映画の reviews_posinega/{movie_id}.xlsx から polarity==1 の文
  負例 (label=0): 好みでない映画の reviews_posinega/{movie_id}.xlsx から polarity==1 の文

モード（実行時に選択）:
  1: 全レビュー文でファインチューニング
     → models/{reviewerID}_allmodel/
  2: TF-IDF 名詞スコア上位 N 件でファインチューニング（nounF.py の手法）
     → models/nounmodels/{reviewerID}/{N}/

実行:
  python finetune_all_reviewers_126.py
"""

import math
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from fugashi import Tagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC
import joblib
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertJapaneseTokenizer,
    Trainer,
    TrainingArguments,
)

# ── パス設定 ──────────────────────────────────────────────────────────────────

BASE_DIR             = Path(__file__).parent
SUMMARY_PATH         = BASE_DIR / "reviewer_summary.xlsx"
POSINEGA_DIR         = BASE_DIR / "reviews_posinega"
MOVIE_DATABASE_DIR   = BASE_DIR / "movie_database"
# MODELS_DIR           = Path(r"C:\Users\Oyabu\GoogleDriveStreaming\マイドライブ\models")
MODELS_DIR           = Path(r"/home/oyabu/GoogleDriveRclone/models")
RANKINGS_DIR         = BASE_DIR / "score_rankings"

BERT_MODEL = "cl-tohoku/bert-base-japanese"

_tagger = Tagger()  # モジュール起動時に1回だけ初期化

# ── Dataset ───────────────────────────────────────────────────────────────────

class MovieReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# ── ユーティリティ ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def clean_output_directory(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ── レビュー読み込み ──────────────────────────────────────────────────────────

def load_positive_sentences(movie_ids: list) -> list[str]:
    """reviews_posinega/{movie_id}.xlsx から polarity==1 の文を収集する。"""
    sentences = []
    for movie_id in movie_ids:
        path = POSINEGA_DIR / f"{movie_id}.xlsx"
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path, header=None)
            pos = df[df.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            sentences.extend([str(s) for s in pos if str(s).strip()])
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")
    return sentences

# ── TF-IDF 名詞スコア（svmmodels と同じ式） ──────────────────────────────────

def extract_reviews_and_scores(
    movie_ids: list,
    all_movie_ids: list | None = None,
) -> tuple[list[tuple[str, str]], dict]:
    """
    reviews_posinega から polarity==1 文を取得し、
    _build_movie_tfidf と同じ式（式5.1, 5.2）で映画単位TF-IDFを計算して返す。
    N・df は all_movie_ids（指定時、好み＋好みでない統合）から計算する。
    戻り値: ([(review, movie_id), ...], {movie_id: {noun: tfidf}})
    """
    reviews_with_movie: list[tuple[str, str]] = []

    for movie_id in movie_ids:
        path = POSINEGA_DIR / f"{movie_id}.xlsx"
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path, header=None)
            pos_reviews = df[df.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            for review in pos_reviews:
                review = str(review).strip()
                if review:
                    reviews_with_movie.append((review, movie_id))
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")

    movie_tfidf = _build_movie_tfidf(movie_ids, all_movie_ids=all_movie_ids)
    return reviews_with_movie, movie_tfidf


def score_reviews(
    reviews_with_movie: list[tuple[str, str]],
    movie_tfidf: dict,
) -> list[tuple[str, float]]:
    """各レビュー文にその映画のTF-IDFスコアの合計を付与する。"""
    result = []
    for review, movie_id in reviews_with_movie:
        tfidf = movie_tfidf.get(movie_id, {})
        s = sum(
            tfidf.get(w.surface, 0.0)
            for w in _tagger(review) if "名詞" in w.feature
        )
        result.append((review, s))
    return result

# ── ランキング保存 ────────────────────────────────────────────────────────────

def save_ranking_xlsx(reviewer_id: int, liked_scored: list, disliked_scored: list):
    """好み・好みでないレビューのスコアランキングを xlsx に保存する。"""
    reviewer_dir = RANKINGS_DIR / str(reviewer_id)
    reviewer_dir.mkdir(parents=True, exist_ok=True)

    liked_df = pd.DataFrame(
        [{"rank": i + 1, "review": text, "score": score}
         for i, (text, score) in enumerate(liked_scored)]
    )
    disliked_df = pd.DataFrame(
        [{"rank": i + 1, "review": text, "score": score}
         for i, (text, score) in enumerate(disliked_scored)]
    )

    out_path = reviewer_dir / "score_ranking.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        liked_df.to_excel(writer, sheet_name="liked", index=False)
        disliked_df.to_excel(writer, sheet_name="disliked", index=False)
    print(f"  [RANKING] 保存: {out_path}")


# ── 映画データベース ──────────────────────────────────────────────────────────

_movie_db_cache: dict | None = None

def _load_movie_db() -> dict:
    global _movie_db_cache
    if _movie_db_cache is None:
        db_path = MOVIE_DATABASE_DIR / "movie_tfidf.joblib"
        print(f"[INFO] 映画データベースを読み込み中: {db_path}")
        _movie_db_cache = joblib.load(db_path)
    return _movie_db_cache


# ── SVM ファインチューニング ──────────────────────────────────────────────────

def _build_movie_tfidf(movie_ids: list, all_movie_ids: list | None = None) -> dict:
    """
    movie_database/movie_tfidf.joblib から指定映画のTF-IDFを返す。
    all_movie_ids は互換性のために残しているが使用しない。
    戻り値: {movie_id: {noun: tfidf_value}}
    """
    db = _load_movie_db()
    return {mid: db[mid] for mid in movie_ids if mid in db}


def _extract_review_vectors(movie_ids: list, movie_tfidf: dict) -> list[dict]:
    """
    各映画の polarity==1 レビュー文を読み込み、
    映画DBのtf·idf値を使って特徴ベクトル（dict形式）に変換する。
    """
    vectors: list[dict] = []
    for movie_id in movie_ids:
        tfidf = movie_tfidf.get(movie_id)
        if tfidf is None:
            continue
        path = POSINEGA_DIR / f"{movie_id}.xlsx"
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path, header=None)
            pos_reviews = df[df.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            for review in pos_reviews:
                review = str(review).strip()
                if not review:
                    continue
                vec = {
                    w.surface: tfidf[w.surface]
                    for w in _tagger(review)
                    if "名詞" in w.feature and w.surface in tfidf
                }
                vectors.append(vec if vec else {"__empty__": 0.0})
        except Exception as e:
            print(f"  [WARN] {movie_id}: {e}")
    return vectors


def finetune_svm(
    reviewer_id:   int,
    liked_vecs:    list[dict],
    disliked_vecs: list[dict],
    min_movie_count: int = 0,
):
    """
    論文 式5.1, 5.2 の映画単位TF-IDF + LinearSVC でレビュワー嗜好を学習する。
    → models/{min_movie_count}/svmmodels/{reviewer_id}/
    """
    model_dir = MODELS_DIR / str(min_movie_count) / "svmmodels" / str(reviewer_id)
    if model_dir.exists():
        print(f"  [SKIP] 既存モデル: {model_dir}")
        return None

    if not liked_vecs or not disliked_vecs:
        print(f"  [SKIP] reviewer {reviewer_id}: 正例または負例が 0 件")
        return None

    print(f"  正例: {len(liked_vecs):,}  負例: {len(disliked_vecs):,}")

    all_vecs   = liked_vecs + disliked_vecs
    all_labels = [1] * len(liked_vecs) + [0] * len(disliked_vecs)

    test_vecs   = all_vecs[:10]
    test_labels = all_labels[:10]

    dv = DictVectorizer()
    X_train = dv.fit_transform(all_vecs)
    X_test  = dv.transform(test_vecs)

    clf = LinearSVC(C=1, max_iter=10000)
    clf.fit(X_train, all_labels)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "svm_model.pkl")
    joblib.dump(dv,  model_dir / "dict_vectorizer.pkl")

    preds = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(test_labels, preds)
    m = {"accuracy": round(acc, 4), "precision": round(float(precision), 4),
         "recall": round(float(recall), 4), "f1": round(float(f1), 4)}
    print(f"  評価結果: {m}")
    return m


# ── BERT ファインチューニング ──────────────────────────────────────────────────

def finetune(
    reviewer_id: int,
    liked_reviews:    list[str],
    disliked_reviews: list[str],
    mode: str,
    tokenizer: BertJapaneseTokenizer,
    min_movie_count: int = 0,
    top_n: int = 0,
    pct: int = 0,
):
    """
    mode: "all"   → 全件使用   → models/{min_movie_count}/allmodels/{reviewerID}/
    mode: "topn"  → 上位N件    → models/{min_movie_count}/nounmodels/{reviewerID}/{N}/
    mode: "pct"   → 上位N%件   → models/{min_movie_count}/nounmodels/{reviewerID}/pct{pct}/
    """
    if not liked_reviews or not disliked_reviews:
        print(f"  [SKIP] reviewer {reviewer_id}: 正例または負例が 0 件")
        return None

    # ── モデル保存先 ──
    if mode == "all":
        model_dir  = str(MODELS_DIR / str(min_movie_count) / "allmodels" / str(reviewer_id))
        output_dir = f"./output/{reviewer_id}_all"
    elif mode == "topn":
        model_dir  = str(MODELS_DIR / str(min_movie_count) / "nounmodels" / str(reviewer_id) / str(top_n))
        output_dir = f"./output/{reviewer_id}_noun{top_n}"
    else:  # pct
        model_dir  = str(MODELS_DIR / str(min_movie_count) / "pctmodels" / str(reviewer_id) / f"pct{pct}")
        output_dir = f"./output/{reviewer_id}_pct{pct}"

    if Path(model_dir).exists():
        print(f"  [SKIP] 既存モデル: {model_dir}")
        return None

    print(f"  正例: {len(liked_reviews):,}  負例: {len(disliked_reviews):,}")

    all_texts  = liked_reviews + disliked_reviews
    all_labels = [1] * len(liked_reviews) + [0] * len(disliked_reviews)

    test_texts  = all_texts[:10]
    test_labels = all_labels[:10]

    train_enc = tokenizer(all_texts,  truncation=True, padding=True, max_length=512)
    test_enc  = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = MovieReviewDataset(train_enc, all_labels)
    test_dataset  = MovieReviewDataset(test_enc,  test_labels)

    set_seed(42)

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 64,
        warmup_steps                = 500,
        weight_decay                = 0.01,
        report_to                   = "none",
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    clean_output_directory(output_dir)

    preds    = trainer.predict(test_dataset)
    eval_res = compute_metrics(preds)
    print(f"  評価結果: {eval_res}")
    return eval_res

# ── レビュー件数テキスト出力 ──────────────────────────────────────────────────

def append_noun_counts(path: Path, reviewer_id: int, liked_count: int, disliked_count: int):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"reviewer_id={reviewer_id}\tliked={liked_count}\tdisliked={disliked_count}\n")


# ── ID パース ─────────────────────────────────────────────────────────────────

def parse_ids(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    # 数値型（例: 527.0）で保存されている場合は整数に変換
    if isinstance(raw, (int, float)):
        return [str(int(raw))]
    return [s.strip() for s in str(raw).split(",") if s.strip().isdigit()]


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    # ── モード選択 ──
    print("=" * 60)
    print("ファインチューニングモードを選択してください")
    print("  1: SVM (linear kernel) でファインチューニング")
    print("  2: 全レビュー文で BERT をファインチューニング")
    print("  3: TF-IDF 名詞スコアで BERT をファインチューニング（nounmodels）")
    print("=" * 60)
    mode_input = input("モード (1, 2, or 3): ").strip()
    while mode_input not in ("1", "2", "3"):
        mode_input = input("1, 2, または 3 を入力してください: ").strip()

    # モード3のサブモード選択
    noun_submode = None
    if mode_input == "3":
        print("-" * 60)
        print("  nounmodels サブモードを選択してください")
        print("  1: N=100~5000（固定件数）")
        print("  2: N=10%~90%（レビュー数に対する割合）→ pctmodels に保存")
        print("-" * 60)
        noun_submode = input("サブモード (1 or 2): ").strip()
        while noun_submode not in ("1", "2"):
            noun_submode = input("1 または 2 を入力してください: ").strip()

    TOP_N_LIST   = list(range(100, 5001, 100))   # 100, 200, ..., 5000
    PERCENT_LIST = list(range(10, 91, 10))        # 10, 20, ..., 90

    # ── 最小映画件数フィルタ ──
    min_n_input = input("好み・好みでない映画がそれぞれ何件以上のレビュワーを対象にしますか？（例: 10）: ").strip()
    while not min_n_input.isdigit() or int(min_n_input) <= 0:
        min_n_input = input("正の整数を入力してください: ").strip()
    min_movie_count = int(min_n_input)

    if mode_input == "1":
        mode_label = "svm"
        mode_desc  = "SVM (linear kernel)"
    elif mode_input == "2":
        mode_label = "all"
        mode_desc  = "全レビュー BERT"
    elif noun_submode == "1":
        mode_label = "topn"
        mode_desc  = "nounmodels モード1: 上位 N 件 BERT (N=100~5000)"
    else:
        mode_label = "pct"
        mode_desc  = "nounmodels モード2: 上位 N% BERT (10%~90%) → pctmodels"

    print(f"\n[INFO] モード: {mode_desc}")
    print(f"[INFO] 対象レビュワー条件: 好み映画 >= {min_movie_count} 件 かつ 好みでない映画 >= {min_movie_count} 件\n")

    # ── BERT モード用トークナイザを1回だけロード ──
    tokenizer = None
    if mode_label != "svm":
        print("[INFO] BertJapaneseTokenizer をロード中...")
        tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL)

    # ── reviewer_summary.xlsx 読み込み ──
    print(f"[INFO] {SUMMARY_PATH.name} を読み込み中...")
    master = pd.read_excel(SUMMARY_PATH, sheet_name="reviewer_master")
    pref   = pd.read_excel(SUMMARY_PATH, sheet_name="reviewer_preference")
    id_to_name: dict = master.set_index("reviewer_id")["reviewer_name"].to_dict()

    MODELS_DIR.mkdir(exist_ok=True)
    RANKINGS_DIR.mkdir(exist_ok=True)

    all_stats    = []
    # stats_path   = BASE_DIR / f"reviewer_stats_{min_movie_count}.xlsx"  # 一時コメントアウト
    all_metrics  = []
    # metrics_path = BASE_DIR / f"train_metrics_{mode_label}_{min_movie_count}.xlsx"  # 一時コメントアウト
    # noun_mode1_counts_path = BASE_DIR / f"noun_mode1_counts_{min_movie_count}.txt"  # 一時コメントアウト
    # noun_mode2_counts_path = BASE_DIR / f"noun_mode2_counts_{min_movie_count}.txt"  # 一時コメントアウト

    def append_and_save_stats(entry: dict):
        all_stats.append(entry)
        # pd.DataFrame(all_stats).to_excel(stats_path, index=False)  # 一時コメントアウト

    def append_and_save_metrics(entry: dict):
        all_metrics.append(entry)
        # pd.DataFrame(all_metrics).to_excel(metrics_path, index=False)  # 一時コメントアウト

    MIN_REVIEWER_ID = 126
    MAX_REVIEWER_ID = 250
    total_reviewers = len(pref)
    for idx, row in pref.iterrows():
        reviewer_id = int(row["reviewer"])
        if reviewer_id < MIN_REVIEWER_ID:
            continue
        if reviewer_id > MAX_REVIEWER_ID:
            print(f"  [STOP] reviewer_id {reviewer_id} が上限 {MAX_REVIEWER_ID} を超えたため終了")
            break
        reviewer_name = id_to_name.get(reviewer_id, str(reviewer_id))

        liked_raw    = row.get("liked_movie_ids",    "")
        disliked_raw = row.get("disliked_movie_ids", "")

        liked_ids    = parse_ids(liked_raw)
        disliked_ids = parse_ids(disliked_raw)

        print(f"[{idx+1}/{total_reviewers}] reviewer {reviewer_id} ({reviewer_name})")
        print(f"  好み映画: {len(liked_ids)} 件  好みでない映画: {len(disliked_ids)} 件")

        if len(liked_ids) < min_movie_count or len(disliked_ids) < min_movie_count:
            print(f"  [SKIP] 映画件数が条件未満 (好み: {len(liked_ids)}, 好みでない: {len(disliked_ids)}, 必要: {min_movie_count})")
            continue

        if mode_label == "svm":
            all_ids        = liked_ids + disliked_ids
            liked_tfidf    = _build_movie_tfidf(liked_ids,    all_movie_ids=all_ids)
            disliked_tfidf = _build_movie_tfidf(disliked_ids, all_movie_ids=all_ids)
            liked_vecs     = _extract_review_vectors(liked_ids,    liked_tfidf)
            disliked_vecs  = _extract_review_vectors(disliked_ids, disliked_tfidf)
            append_and_save_stats({
                "reviewer_id":           reviewer_id,
                "reviewer_name":         reviewer_name,
                "liked_movie_count":     len(liked_ids),
                "disliked_movie_count":  len(disliked_ids),
                "liked_review_count":    len(liked_vecs),
                "disliked_review_count": len(disliked_vecs),
            })
            m = finetune_svm(
                reviewer_id     = reviewer_id,
                liked_vecs      = liked_vecs,
                disliked_vecs   = disliked_vecs,
                min_movie_count = min_movie_count,
            )
            if m:
                append_and_save_metrics({"reviewer_id": reviewer_id, "mode": "svm", **m})
        elif mode_label == "all":
            liked_reviews    = load_positive_sentences(liked_ids)
            disliked_reviews = load_positive_sentences(disliked_ids)
            append_and_save_stats({
                "reviewer_id":           reviewer_id,
                "reviewer_name":         reviewer_name,
                "liked_movie_count":     len(liked_ids),
                "disliked_movie_count":  len(disliked_ids),
                "liked_review_count":    len(liked_reviews),
                "disliked_review_count": len(disliked_reviews),
            })
            m = finetune(
                reviewer_id      = reviewer_id,
                liked_reviews    = liked_reviews,
                disliked_reviews = disliked_reviews,
                mode             = mode_label,
                tokenizer        = tokenizer,
                min_movie_count  = min_movie_count,
            )
            if m:
                append_and_save_metrics({"reviewer_id": reviewer_id, "mode": "all", **m})
        elif mode_label == "topn":
            # TF-IDF スコアを一度だけ計算してN値ごとに再利用
            all_ids = liked_ids + disliked_ids
            liked_reviews_with_movie,    liked_tfidf    = extract_reviews_and_scores(liked_ids,    all_movie_ids=all_ids)
            disliked_reviews_with_movie, disliked_tfidf = extract_reviews_and_scores(disliked_ids, all_movie_ids=all_ids)

            append_and_save_stats({
                "reviewer_id":           reviewer_id,
                "reviewer_name":         reviewer_name,
                "liked_movie_count":     len(liked_ids),
                "disliked_movie_count":  len(disliked_ids),
                "liked_review_count":    len(liked_reviews_with_movie),
                "disliked_review_count": len(disliked_reviews_with_movie),
            })

            liked_scored    = score_reviews(liked_reviews_with_movie,    liked_tfidf)
            disliked_scored = score_reviews(disliked_reviews_with_movie, disliked_tfidf)

            liked_scored.sort(key=lambda x: x[1],    reverse=True)
            disliked_scored.sort(key=lambda x: x[1], reverse=True)

            # save_ranking_xlsx(reviewer_id, liked_scored, disliked_scored)  # 一時コメントアウト

            top_n_step = TOP_N_LIST[1] - TOP_N_LIST[0] if len(TOP_N_LIST) > 1 else TOP_N_LIST[0]
            for top_n in TOP_N_LIST:
                if min(len(liked_scored), len(disliked_scored)) <= top_n - top_n_step:
                    print(f"  [STOP] N={top_n} に必要な文数が不足 "
                          f"(正例: {len(liked_scored)}, 負例: {len(disliked_scored)})")
                    break
                print(f"  [N={top_n}]")
                m = finetune(
                    reviewer_id      = reviewer_id,
                    liked_reviews    = [r for r, _ in liked_scored[:top_n]],
                    disliked_reviews = [r for r, _ in disliked_scored[:top_n]],
                    mode             = "topn",
                    tokenizer        = tokenizer,
                    min_movie_count  = min_movie_count,
                    top_n            = top_n,
                )
                if m:
                    append_and_save_metrics({"reviewer_id": reviewer_id, "mode": "topn", "top_n": top_n, **m})

            # append_noun_counts(noun_mode1_counts_path, reviewer_id,  # 一時コメントアウト
            #                    len(liked_scored), len(disliked_scored))

        elif mode_label == "pct":
            # TF-IDF スコアを一度だけ計算してパーセンテージごとに再利用
            all_ids = liked_ids + disliked_ids
            liked_reviews_with_movie,    liked_tfidf    = extract_reviews_and_scores(liked_ids,    all_movie_ids=all_ids)
            disliked_reviews_with_movie, disliked_tfidf = extract_reviews_and_scores(disliked_ids, all_movie_ids=all_ids)

            append_and_save_stats({
                "reviewer_id":           reviewer_id,
                "reviewer_name":         reviewer_name,
                "liked_movie_count":     len(liked_ids),
                "disliked_movie_count":  len(disliked_ids),
                "liked_review_count":    len(liked_reviews_with_movie),
                "disliked_review_count": len(disliked_reviews_with_movie),
            })

            liked_scored    = score_reviews(liked_reviews_with_movie,    liked_tfidf)
            disliked_scored = score_reviews(disliked_reviews_with_movie, disliked_tfidf)

            liked_scored.sort(key=lambda x: x[1],    reverse=True)
            disliked_scored.sort(key=lambda x: x[1], reverse=True)

            # save_ranking_xlsx(reviewer_id, liked_scored, disliked_scored)  # 一時コメントアウト

            min_class_count = min(len(liked_scored), len(disliked_scored))
            for pct in PERCENT_LIST:
                n = max(1, int(min_class_count * pct / 100))
                print(f"  [{pct}%] 基準件数: {min_class_count:,} → 正例: {n:,} 件  負例: {n:,} 件")
                m = finetune(
                    reviewer_id      = reviewer_id,
                    liked_reviews    = [r for r, _ in liked_scored[:n]],
                    disliked_reviews = [r for r, _ in disliked_scored[:n]],
                    mode             = "pct",
                    tokenizer        = tokenizer,
                    min_movie_count  = min_movie_count,
                    pct              = pct,
                )
                if m:
                    append_and_save_metrics({"reviewer_id": reviewer_id, "mode": "pct", "pct": pct, **m})

            # append_noun_counts(noun_mode2_counts_path, reviewer_id,  # 一時コメントアウト
            #                    len(liked_scored), len(disliked_scored))

    print("\n[DONE] 全レビュワーのファインチューニングが完了しました。")


if __name__ == "__main__":
    main()
