# count_finetune_reviews.py
# -*- coding: utf-8 -*-
"""
reviewer_summary.xlsx を参照し、各レビュワーのファインチューニングに用いる
正例・負例レビュー文数を集計して xlsx に出力する。実際の学習は行わない。

出力:
  all / svm モード  → review_counts_{mode}_{min_movie_count}.xlsx
                      （レビュワーごとに1行）
  topn モード       → review_counts_topn_{min_movie_count}.xlsx
                      （レビュワー × N ごとに1行）
  pct モード        → review_counts_pct_{min_movie_count}.xlsx
                      （レビュワー × % ごとに1行）
"""

import math
from pathlib import Path

import pandas as pd
from fugashi import Tagger
import joblib

# ── パス設定 ──────────────────────────────────────────────────────────────────

BASE_DIR           = Path(__file__).parent
SUMMARY_PATH       = BASE_DIR / "reviewer_summary.xlsx"
POSINEGA_DIR       = BASE_DIR / "reviews_posinega"
MOVIE_DATABASE_DIR = BASE_DIR / "movie_database"

_tagger = Tagger()

# ── レビュー読み込み ──────────────────────────────────────────────────────────

def load_positive_sentences(movie_ids: list) -> list[str]:
    sentences = []
    for movie_id in movie_ids:
        path = POSINEGA_DIR / f"{movie_id}.xlsx"
        if not path.exists():
            continue
        try:
            df  = pd.read_excel(path, header=None)
            pos = df[df.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            sentences.extend([str(s) for s in pos if str(s).strip()])
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")
    return sentences

# ── 映画データベース（SVM モード用） ─────────────────────────────────────────

_movie_db_cache: dict | None = None

def _load_movie_db() -> dict:
    global _movie_db_cache
    if _movie_db_cache is None:
        db_path = MOVIE_DATABASE_DIR / "movie_tfidf.joblib"
        print(f"[INFO] 映画データベースを読み込み中: {db_path}")
        _movie_db_cache = joblib.load(db_path)
    return _movie_db_cache

def _build_movie_tfidf(movie_ids: list, all_movie_ids: list | None = None) -> dict:
    db = _load_movie_db()
    return {mid: db[mid] for mid in movie_ids if mid in db}

def _count_review_vectors(movie_ids: list, movie_tfidf: dict) -> int:
    """SVM の実際の学習データ数（TF-IDF ベクトル化後の件数）を返す。"""
    count = 0
    for movie_id in movie_ids:
        tfidf = movie_tfidf.get(movie_id)
        if tfidf is None:
            continue
        path = POSINEGA_DIR / f"{movie_id}.xlsx"
        if not path.exists():
            continue
        try:
            df          = pd.read_excel(path, header=None)
            pos_reviews = df[df.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            for review in pos_reviews:
                review = str(review).strip()
                if review:
                    count += 1
        except Exception as e:
            print(f"  [WARN] {movie_id}: {e}")
    return count

# ── ID パース ─────────────────────────────────────────────────────────────────

def parse_ids(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, (int, float)):
        return [str(int(raw))]
    return [s.strip() for s in str(raw).split(",") if s.strip().isdigit()]

# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("レビュー文数集計モードを選択してください")
    print("  1: SVM (linear kernel)")
    print("  2: 全レビュー文 BERT")
    print("  3: TF-IDF 名詞スコア BERT（nounmodels）")
    print("=" * 60)
    mode_input = input("モード (1, 2, or 3): ").strip()
    while mode_input not in ("1", "2", "3"):
        mode_input = input("1, 2, または 3 を入力してください: ").strip()

    noun_submode = None
    if mode_input == "3":
        print("-" * 60)
        print("  サブモードを選択してください")
        print("  1: N=100~5000（固定件数）")
        print("  2: N=10%~90%（レビュー数に対する割合）")
        print("-" * 60)
        noun_submode = input("サブモード (1 or 2): ").strip()
        while noun_submode not in ("1", "2"):
            noun_submode = input("1 または 2 を入力してください: ").strip()

    TOP_N_LIST   = list(range(100, 5001, 100))   # 100, 200, ..., 5000
    PERCENT_LIST = list(range(10, 91, 10))        # 10, 20, ..., 90

    min_n_input = input("好み・好みでない映画がそれぞれ何件以上のレビュワーを対象にしますか？（例: 10）: ").strip()
    while not min_n_input.isdigit() or int(min_n_input) <= 0:
        min_n_input = input("正の整数を入力してください: ").strip()
    min_movie_count = int(min_n_input)

    if mode_input == "1":
        mode_label = "svm"
    elif mode_input == "2":
        mode_label = "all"
    elif noun_submode == "1":
        mode_label = "topn"
    else:
        mode_label = "pct"

    print(f"\n[INFO] モード: {mode_label}")
    print(f"[INFO] 対象レビュワー条件: 好み映画 >= {min_movie_count} 件 かつ 好みでない映画 >= {min_movie_count} 件\n")

    master     = pd.read_excel(SUMMARY_PATH, sheet_name="reviewer_master")
    pref       = pd.read_excel(SUMMARY_PATH, sheet_name="reviewer_preference")
    id_to_name = master.set_index("reviewer_id")["reviewer_name"].to_dict()

    all_rows      = []
    out_path      = BASE_DIR / f"review_counts_{mode_label}_{min_movie_count}.xlsx"
    MAX_REVIEWER_ID = 250
    total         = len(pref)

    def save():
        pd.DataFrame(all_rows).to_excel(out_path, index=False)

    for idx, row in pref.iterrows():
        reviewer_id = int(row["reviewer"])
        if reviewer_id > MAX_REVIEWER_ID:
            print(f"  [STOP] reviewer_id {reviewer_id} が上限 {MAX_REVIEWER_ID} を超えたため終了")
            break
        reviewer_name = id_to_name.get(reviewer_id, str(reviewer_id))

        liked_ids    = parse_ids(row.get("liked_movie_ids",    ""))
        disliked_ids = parse_ids(row.get("disliked_movie_ids", ""))

        print(f"[{idx+1}/{total}] reviewer {reviewer_id} ({reviewer_name})")
        print(f"  好み映画: {len(liked_ids)} 件  好みでない映画: {len(disliked_ids)} 件")

        if len(liked_ids) < min_movie_count or len(disliked_ids) < min_movie_count:
            print(f"  [SKIP] 映画件数が条件未満 (好み: {len(liked_ids)}, 好みでない: {len(disliked_ids)}, 必要: {min_movie_count})")
            continue

        base = {
            "reviewer_id":          reviewer_id,
            "reviewer_name":        reviewer_name,
            "liked_movie_count":    len(liked_ids),
            "disliked_movie_count": len(disliked_ids),
        }

        if mode_label == "svm":
            all_ids        = liked_ids + disliked_ids
            liked_tfidf    = _build_movie_tfidf(liked_ids,    all_movie_ids=all_ids)
            disliked_tfidf = _build_movie_tfidf(disliked_ids, all_movie_ids=all_ids)
            liked_count    = _count_review_vectors(liked_ids,    liked_tfidf)
            disliked_count = _count_review_vectors(disliked_ids, disliked_tfidf)
            print(f"  正例: {liked_count:,}  負例: {disliked_count:,}")
            all_rows.append({**base,
                             "liked_review_count":    liked_count,
                             "disliked_review_count": disliked_count})

        elif mode_label == "all":
            liked_count    = len(load_positive_sentences(liked_ids))
            disliked_count = len(load_positive_sentences(disliked_ids))
            print(f"  正例: {liked_count:,}  負例: {disliked_count:,}")
            all_rows.append({**base,
                             "liked_review_count":    liked_count,
                             "disliked_review_count": disliked_count})

        elif mode_label == "topn":
            liked_count    = len(load_positive_sentences(liked_ids))
            disliked_count = len(load_positive_sentences(disliked_ids))
            top_n_step     = TOP_N_LIST[1] - TOP_N_LIST[0] if len(TOP_N_LIST) > 1 else TOP_N_LIST[0]
            for top_n in TOP_N_LIST:
                if min(liked_count, disliked_count) <= top_n - top_n_step:
                    print(f"  [STOP] N={top_n} 以降は文数不足 "
                          f"(正例: {liked_count}, 負例: {disliked_count})")
                    break
                n_liked    = min(top_n, liked_count)
                n_disliked = min(top_n, disliked_count)
                all_rows.append({**base,
                                 "liked_review_count_total":    liked_count,
                                 "disliked_review_count_total": disliked_count,
                                 "top_n":           top_n,
                                 "liked_n_used":    n_liked,
                                 "disliked_n_used": n_disliked})

        elif mode_label == "pct":
            liked_count    = len(load_positive_sentences(liked_ids))
            disliked_count = len(load_positive_sentences(disliked_ids))
            min_class      = min(liked_count, disliked_count)
            for pct in PERCENT_LIST:
                n = max(1, int(min_class * pct / 100))
                print(f"  [{pct}%] 基準: {min_class:,} → 正例・負例ともに {n:,} 件")
                all_rows.append({**base,
                                 "liked_review_count_total":    liked_count,
                                 "disliked_review_count_total": disliked_count,
                                 "min_class_count": min_class,
                                 "pct":             pct,
                                 "n_used":          n})

        save()

    print(f"\n[DONE] 保存完了: {out_path}")


if __name__ == "__main__":
    main()