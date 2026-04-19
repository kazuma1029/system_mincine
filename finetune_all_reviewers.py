# finetune_all_reviewers.py
# -*- coding: utf-8 -*-
"""
全レビュワーに対して reviewer_summary.xlsx から好み映画・好みでない映画を参照し、
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
  python finetune_all_reviewers.py
"""

import os
import shutil
from collections import Counter
from math import log
from pathlib import Path

import pandas as pd
import torch
from fugashi import Tagger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertJapaneseTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ── パス設定 ──────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
SUMMARY_PATH     = BASE_DIR / "reviewer_summary.xlsx"
POSINEGA_DIR     = BASE_DIR / "reviews_posinega"
MODELS_DIR       = BASE_DIR / "models"
RANKINGS_DIR     = BASE_DIR / "score_rankings"
# LOG_DIR          = "C:/Users/kazuma/logs"
LOG_DIR          = "C:/Users/Oyabu/research/logs"

BERT_MODEL = "cl-tohoku/bert-base-japanese"

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

# ── TF-IDF 名詞スコア（nounF.py の手法） ─────────────────────────────────────

def calculate_tfidf(all_noun_lists: list, total_movies: int):
    term_movie_count = Counter()
    tf_values        = []

    for noun_list in all_noun_lists:
        doc_terms = Counter()
        for nouns_str in noun_list:
            doc_terms.update(nouns_str.split())
        tf_values.append(doc_terms)
        for term in set(doc_terms):
            term_movie_count[term] += 1

    idf = {t: log(total_movies / (1 + c)) for t, c in term_movie_count.items()}
    tfidf_scores = [{t: v * idf[t] for t, v in tf.items()} for tf in tf_values]
    return tfidf_scores


def extract_reviews_and_scores(movie_ids: list) -> tuple[list[str], Counter]:
    """
    reviews_posinega から polarity==1 文を取得し、
    TF-IDF 名詞スコア辞書を返す。
    """
    tagger      = Tagger()
    all_reviews = []
    all_nouns   = []      # 映画ごとの名詞リスト

    for movie_id in movie_ids:
        path = POSINEGA_DIR / f"{movie_id}.xlsx"
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path, header=None)
            pos_reviews = df[df.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            movie_nouns = []
            for review in pos_reviews:
                review = str(review).strip()
                if not review:
                    continue
                nouns_str = " ".join(
                    w.surface for w in tagger(review) if "名詞" in w.feature
                )
                if nouns_str:
                    all_reviews.append(review)
                    movie_nouns.append(nouns_str)
            if movie_nouns:
                all_nouns.append(movie_nouns)
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")

    total_scores: Counter = Counter()
    if all_nouns:
        tfidf_list = calculate_tfidf(all_nouns, total_movies=len(movie_ids))
        for doc_scores in tfidf_list:
            total_scores.update(doc_scores)

    return all_reviews, total_scores


def score_reviews(reviews: list[str], total_scores: Counter) -> list[tuple[str, float]]:
    """各レビュー文に名詞TF-IDFスコアの合計を付与する。"""
    tagger = Tagger()
    result = []
    for review in reviews:
        s = sum(
            total_scores.get(w.surface, 0)
            for w in tagger(review) if "名詞" in w.feature
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


# ── BERT ファインチューニング ──────────────────────────────────────────────────

def finetune(
    reviewer_id: int,
    liked_reviews:    list[str],
    disliked_reviews: list[str],
    mode: str,
    min_movie_count: int = 0,
    top_n: int = 0,
):
    """
    mode: "all"   → 全件使用   → models/{min_movie_count}/allmodels/{reviewerID}/
    mode: "topn"  → 上位N件    → models/{min_movie_count}/nounmodels/{reviewerID}/{N}/
    """
    if not liked_reviews or not disliked_reviews:
        print(f"  [SKIP] reviewer {reviewer_id}: 正例または負例が 0 件")
        return

    # ── モデル保存先 ──
    if mode == "all":
        model_dir  = str(MODELS_DIR / str(min_movie_count) / "allmodels" / str(reviewer_id))
        output_dir = f"./output/{reviewer_id}_all"
    else:
        model_dir  = str(MODELS_DIR / str(min_movie_count) / "nounmodels" / str(reviewer_id) / str(top_n))
        output_dir = f"./output/{reviewer_id}_noun{top_n}"

    if Path(model_dir).exists():
        print(f"  [SKIP] 既存モデル: {model_dir}")
        return

    print(f"  正例: {len(liked_reviews):,}  負例: {len(disliked_reviews):,}")

    all_texts  = liked_reviews + disliked_reviews
    all_labels = [1] * len(liked_reviews) + [0] * len(disliked_reviews)

    test_texts  = all_texts[:10]
    test_labels = all_labels[:10]

    set_seed(42)

    tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL)
    model     = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)

    # padding=False でトークナイズし、バッチ単位で動的パディング（メモリ節約）
    train_enc = tokenizer(all_texts,  truncation=True, padding=False, max_length=512)
    test_enc  = tokenizer(test_texts, truncation=True, padding=False, max_length=512)

    train_dataset = MovieReviewDataset(train_enc, all_labels)
    test_dataset  = MovieReviewDataset(test_enc,  test_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 64,
        warmup_steps                = 500,
        weight_decay                = 0.01,
        logging_dir                 = LOG_DIR,
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
        data_collator   = data_collator,
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    clean_output_directory(output_dir)

    preds      = trainer.predict(test_dataset)
    eval_res   = compute_metrics(preds)
    print(f"  評価結果: {eval_res}")

# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    # ── モード選択 ──
    print("=" * 60)
    print("ファインチューニングモードを選択してください")
    print("  1: 全レビュー文でファインチューニング")
    print("  2: TF-IDF 名詞スコア上位 N 件でファインチューニング")
    print("=" * 60)
    mode_input = input("モード (1 or 2): ").strip()
    while mode_input not in ("1", "2"):
        mode_input = input("1 または 2 を入力してください: ").strip()

    TOP_N_LIST = list(range(100, 5001, 100))  # 100, 200, ..., 5000

    # ── 最小映画件数フィルタ ──
    min_n_input = input("好み・好みでない映画がそれぞれ何件以上のレビュワーを対象にしますか？（例: 10）: ").strip()
    while not min_n_input.isdigit() or int(min_n_input) <= 0:
        min_n_input = input("正の整数を入力してください: ").strip()
    min_movie_count = int(min_n_input)

    mode_label = "all" if mode_input == "1" else "topn"
    print(f"\n[INFO] モード: {'全レビュー' if mode_input == '1' else f'上位 N 件 (N={TOP_N_LIST})'}")
    print(f"[INFO] 対象レビュワー条件: 好み映画 >= {min_movie_count} 件 かつ 好みでない映画 >= {min_movie_count} 件\n")

    # ── reviewer_summary.xlsx 読み込み ──
    print(f"[INFO] {SUMMARY_PATH.name} を読み込み中...")
    master = pd.read_excel(SUMMARY_PATH, sheet_name="reviewer_master")
    pref   = pd.read_excel(SUMMARY_PATH, sheet_name="reviewer_preference")
    id_to_name: dict = master.set_index("reviewer_id")["reviewer_name"].to_dict()

    MODELS_DIR.mkdir(exist_ok=True)
    RANKINGS_DIR.mkdir(exist_ok=True)

    all_stats = []
    total_reviewers = len(pref)
    for idx, row in pref.iterrows():
        reviewer_id = int(row["reviewer"])
        reviewer_name = id_to_name.get(reviewer_id, str(reviewer_id))

        liked_raw    = row.get("liked_movie_ids",    "")
        disliked_raw = row.get("disliked_movie_ids", "")

        def parse_ids(raw) -> list[str]:
            import math
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                return []
            # 数値型（例: 527.0）で保存されている場合は整数に変換
            if isinstance(raw, (int, float)):
                return [str(int(raw))]
            return [s.strip() for s in str(raw).split(",") if s.strip().isdigit()]

        liked_ids    = parse_ids(liked_raw)
        disliked_ids = parse_ids(disliked_raw)

        print(f"[{idx+1}/{total_reviewers}] reviewer {reviewer_id} ({reviewer_name})")
        print(f"  好み映画: {len(liked_ids)} 件  好みでない映画: {len(disliked_ids)} 件")

        if len(liked_ids) < min_movie_count or len(disliked_ids) < min_movie_count:
            print(f"  [SKIP] 映画件数が条件未満 (好み: {len(liked_ids)}, 好みでない: {len(disliked_ids)}, 必要: {min_movie_count})")
            continue

        if mode_label == "all":
            liked_reviews    = load_positive_sentences(liked_ids)
            disliked_reviews = load_positive_sentences(disliked_ids)
            all_stats.append({
                "reviewer_id":           reviewer_id,
                "reviewer_name":         reviewer_name,
                "liked_movie_count":     len(liked_ids),
                "disliked_movie_count":  len(disliked_ids),
                "liked_review_count":    len(liked_reviews),
                "disliked_review_count": len(disliked_reviews),
            })
            finetune(
                reviewer_id      = reviewer_id,
                liked_reviews    = liked_reviews,
                disliked_reviews = disliked_reviews,
                mode             = mode_label,
                min_movie_count  = min_movie_count,
            )
        else:
            # TF-IDF スコアを一度だけ計算してN値ごとに再利用
            liked_reviews_all,    liked_scores    = extract_reviews_and_scores(liked_ids)
            disliked_reviews_all, disliked_scores = extract_reviews_and_scores(disliked_ids)

            all_stats.append({
                "reviewer_id":           reviewer_id,
                "reviewer_name":         reviewer_name,
                "liked_movie_count":     len(liked_ids),
                "disliked_movie_count":  len(disliked_ids),
                "liked_review_count":    len(liked_reviews_all),
                "disliked_review_count": len(disliked_reviews_all),
            })

            liked_scored    = score_reviews(liked_reviews_all,    liked_scores)
            disliked_scored = score_reviews(disliked_reviews_all, disliked_scores)

            liked_scored.sort(key=lambda x: x[1],    reverse=True)
            disliked_scored.sort(key=lambda x: x[1], reverse=True)

            save_ranking_xlsx(reviewer_id, liked_scored, disliked_scored)

            for top_n in TOP_N_LIST:
                if len(liked_scored) < top_n and len(disliked_scored) < top_n:
                    print(f"  [STOP] N={top_n} に必要な文数が不足 "
                          f"(正例: {len(liked_scored)}, 負例: {len(disliked_scored)})")
                    break
                print(f"  [N={top_n}]")
                finetune(
                    reviewer_id      = reviewer_id,
                    liked_reviews    = [r for r, _ in liked_scored[:top_n]],
                    disliked_reviews = [r for r, _ in disliked_scored[:top_n]],
                    mode             = mode_label,
                    min_movie_count  = min_movie_count,
                    top_n            = top_n,
                )

    # ── 全レビュワー統計を xlsx に保存 ──
    if all_stats:
        stats_path = BASE_DIR / f"reviewer_stats_{min_movie_count}.xlsx"
        pd.DataFrame(all_stats).to_excel(stats_path, index=False)
        print(f"[INFO] レビュワー統計を保存: {stats_path}")

    print("\n[DONE] 全レビュワーのファインチューニングが完了しました。")


if __name__ == "__main__":
    main()
