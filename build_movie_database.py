# build_movie_database.py
# -*- coding: utf-8 -*-
"""
reviews/{movie_id}.xlsx の全レビューから映画データベースを構築する。

論文 式5.1, 5.2 に従って映画単位のTF-IDFを計算する。
  tf(t, d)  = n_{t,d} / Σ_{s∈d} n_{s,d}
              d = 映画の全レビューを結合した文書（polarity フィルタなし）
  idf(t)    = log₂(N / df(t)) + 1
              N   = 名詞データのある映画の総数（グローバル）
              df(t)= 名詞 t が出現する映画の数（グローバル）

出力:
  movie_database/movie_tfidf.joblib   {movie_id: {noun: tfidf}} の辞書
  movie_database/build_stats.txt      統計サマリ

使い方:
  c:/Users/Kazuma/anaconda3/python.exe build_movie_database.py
"""

import joblib
from collections import Counter
from math import log
from pathlib import Path

import pandas as pd
from fugashi import Tagger

# ── パス設定 ──────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
REVIEWS_DIR = BASE_DIR / "reviews"
OUTPUT_DIR  = BASE_DIR / "movie_database"

# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    tagger = Tagger()

    # ── Step 1: 全映画の名詞出現回数を集計 ──────────────────────────────────
    files = sorted(
        REVIEWS_DIR.glob("*.xlsx"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    total_files = len(files)
    print(f"[INFO] 対象映画ファイル数: {total_files:,}")

    movie_noun_counts: dict[str, Counter] = {}
    skipped = 0

    for i, path in enumerate(files, 1):
        if i == 1 or i % 1000 == 0:
            print(f"[INFO] {i:,}/{total_files:,}: 名詞集計中 ...")

        movie_id = path.stem
        try:
            df = pd.read_excel(path, dtype={"review_text": str})
            texts = df["review_text"].dropna().tolist()
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")
            skipped += 1
            continue

        noun_count: Counter = Counter()
        for text in texts:
            text = str(text).strip()
            if not text:
                continue
            for w in tagger(text):
                if "名詞" in w.feature:
                    noun_count[w.surface] += 1

        if noun_count:
            movie_noun_counts[movie_id] = noun_count

    N = len(movie_noun_counts)
    print(f"[INFO] 名詞データのある映画数 N = {N:,}  (スキップ: {skipped:,})")

    # ── Step 2: df(t) を計算（全映画横断） ──────────────────────────────────
    print("[INFO] df(t) を計算中 ...")
    df_count: Counter = Counter()
    for nc in movie_noun_counts.values():
        for noun in nc:
            df_count[noun] += 1

    vocab_size = len(df_count)
    print(f"[INFO] 異なり名詞数（語彙サイズ）: {vocab_size:,}")

    # ── Step 3: idf(t) = log₂(N / df(t)) + 1 ───────────────────────────────
    idf: dict[str, float] = {
        noun: log(N / cnt, 2) + 1
        for noun, cnt in df_count.items()
    }

    # ── Step 4: tf(t, d) × idf(t) を映画ごとに算出 ──────────────────────────
    print("[INFO] TF-IDF を計算中 ...")
    movie_tfidf: dict[str, dict[str, float]] = {}
    for movie_id, nc in movie_noun_counts.items():
        total_nouns = sum(nc.values())
        movie_tfidf[movie_id] = {
            noun: (cnt / total_nouns) * idf[noun]
            for noun, cnt in nc.items()
        }

    # ── Step 5: 保存 ─────────────────────────────────────────────────────────
    db_path = OUTPUT_DIR / "movie_tfidf.joblib"
    joblib.dump(movie_tfidf, db_path, compress=3)
    print(f"[DONE] 保存: {db_path}")

    # 統計サマリ
    stats_path = OUTPUT_DIR / "build_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"対象映画ファイル数:        {total_files:,}\n")
        f.write(f"名詞データのある映画数 N:  {N:,}\n")
        f.write(f"スキップ数:                {skipped:,}\n")
        f.write(f"語彙サイズ:                {vocab_size:,}\n")
    print(f"[INFO] 統計保存: {stats_path}")
    print(f"\n  対象映画数:  {total_files:,}")
    print(f"  有効映画数:  {N:,}")
    print(f"  語彙サイズ:  {vocab_size:,}")


if __name__ == "__main__":
    main()
