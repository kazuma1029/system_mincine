# classify_experiment.py
# -*- coding: utf-8 -*-
"""
experiment_all/{reviewer_id}/{movie_id}.xlsx のレビューを各モデルで分類し、
accuracy / precision / recall / F-measure をレビュワーごとに xlsx 出力する。

モード:
  1: SVM (svmmodels)                          → results/results_svmmodels_{min_n}.xlsx
  2: 全レビュー BERT (allmodels)              → results/results_allmodels_{min_n}.xlsx
  3: TF-IDF 上位N件 BERT (nounmodels)        → results/results_nounmodels_{min_n}.xlsx
  4: TF-IDF 上位N% BERT (pctmodels)          → results/results_pctmodels_{min_n}.xlsx

ディレクトリ構成（想定）:
  experiment_all/{reviewer_id}/{movie_id}.xlsx
  {MODELS_DIR}/{min_n}/svmmodels/{reviewer_id}/
  {MODELS_DIR}/{min_n}/allmodels/{reviewer_id}/
  {MODELS_DIR}/{min_n}/nounmodels/{reviewer_id}/{N}/
  {MODELS_DIR}/{min_n}/pctmodels/{reviewer_id}/pct{pct}/
"""

import random
from pathlib import Path

import joblib
import pandas as pd
import torch
from fugashi import Tagger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertJapaneseTokenizer

# ── パス設定（環境に合わせて変更） ────────────────────────────────────────────

BASE_DIR           = Path(__file__).parent
EXPERIMENT_DIR     = BASE_DIR / "experiment_all"
MOVIE_DATABASE_DIR = BASE_DIR / "movie_database"
MODELS_DIR         = Path(r"/home/oyabu/GoogleDriveRclone/models")
OUTPUT_DIR         = BASE_DIR / "results"

BERT_MODEL = "cl-tohoku/bert-base-japanese"

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
    return {
        "accuracy":  round(acc, 4),
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f1, 4),
        "true_pos":  sum(1 for l in y_true if l == 1),
        "true_neg":  sum(1 for l in y_true if l == 0),
        "pred_pos":  sum(1 for l in y_pred if l == 1),
        "pred_neg":  sum(1 for l in y_pred if l == 0),
    }


# ── 増分保存ヘルパー ──────────────────────────────────────────────────────────

def _load_existing_df(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            if path.suffix == ".xlsx":
                return pd.read_excel(path)
            else:
                return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            pass
    return pd.DataFrame()


def _save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def _append_detail_csv(df: pd.DataFrame, path: Path):
    """detail用CSVに新規行のみ追記する（全体を読み込んで書き直さない）。"""
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", index=False, encoding="utf-8-sig",
              header=not path.exists())


# ── BERT 推論 ─────────────────────────────────────────────────────────────────

_tokenizer_cache: BertJapaneseTokenizer | None = None

def _get_tokenizer() -> BertJapaneseTokenizer:
    """トークナイザは全モデル共通(BERT_MODEL由来)なので一度だけロードする。"""
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = BertJapaneseTokenizer.from_pretrained(BERT_MODEL)
    return _tokenizer_cache


def predict_bert(texts: list[str], model_dir: Path) -> list[int]:
    tokenizer = _get_tokenizer()
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


# ── 映画データベース ──────────────────────────────────────────────────────────

_movie_db_cache: dict | None = None

def _load_movie_db() -> dict:
    global _movie_db_cache
    if _movie_db_cache is None:
        db_path = MOVIE_DATABASE_DIR / "movie_tfidf.joblib"
        print(f"[INFO] 映画データベースを読み込み中: {db_path}")
        _movie_db_cache = joblib.load(db_path)
    return _movie_db_cache


# ── SVM 推論 ──────────────────────────────────────────────────────────────────

def _build_tfidf_from_experiment(reviewer_id: int) -> dict:
    """
    movie_database/movie_tfidf.joblib から映画単位のTF-IDFを返す。
    reviewer_id は互換性のために残しているが使用しない。
    戻り値: {movie_id: {noun: tfidf_value}}
    """
    return _load_movie_db()


_tagger_cache: Tagger | None = None

def _get_tagger() -> Tagger:
    global _tagger_cache
    if _tagger_cache is None:
        _tagger_cache = Tagger()
    return _tagger_cache


def predict_svm(texts: list[str], movie_ids: list[str],
                reviewer_id: int, model_dir: Path) -> list[int]:
    """
    映画単位TF-IDFで特徴ベクトルを構築し、保存済み LinearSVC で予測する。
    """
    clf = joblib.load(model_dir / "svm_model.pkl")
    dv  = joblib.load(model_dir / "dict_vectorizer.pkl")
    tagger = _get_tagger()

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

def evaluate_allmodels(min_n: int, reviewer_ids: list[int],
                       experiment_data: dict[int, tuple],
                       out_summary: Path, out_detail: Path) -> None:
    model_base = MODELS_DIR / str(min_n) / "allmodels"
    if not model_base.exists():
        return

    df_summary = _load_existing_df(out_summary)
    done_ids   = set(df_summary["reviewer_id"].tolist()) if not df_summary.empty else set()

    for rid in reviewer_ids:
        if rid in done_ids:
            print(f"  [SKIP] reviewer {rid}: 既存結果あり")
            continue
        model_dir = model_base / str(rid)
        if not model_dir.exists():
            continue
        if rid not in experiment_data:
            continue
        try:
            texts, ratings, movie_ids = experiment_data[rid]
            labels = make_labels(ratings)
            preds  = predict_bert(texts, model_dir)
            m      = compute_metrics(labels, preds)
            df_summary = pd.concat([df_summary, pd.DataFrame([{"reviewer_id": rid, **m}])],
                                   ignore_index=True)
            detail_df = pd.DataFrame([
                {"reviewer_id": rid, "movie_id": mid, "review": t,
                 "rating": r, "true_label": tl, "pred_label": pl}
                for t, mid, r, tl, pl in zip(texts, movie_ids, ratings, labels, preds)
            ])
            _save_df(df_summary, out_summary)
            _append_detail_csv(detail_df, out_detail)
            print(f"  reviewer {rid}: {m}")
        except Exception as e:
            print(f"  [WARN] reviewer {rid}: {e}")


def evaluate_nounmodels(min_n: int, reviewer_ids: list[int],
                        experiment_data: dict[int, tuple],
                        out_summary: Path, out_detail: Path) -> None:
    model_base = MODELS_DIR / str(min_n) / "nounmodels"
    if not model_base.exists():
        return

    df_summary = _load_existing_df(out_summary)
    done_keys  = (set(zip(df_summary["reviewer_id"], df_summary["top_n"]))
                  if not df_summary.empty else set())

    for rid in reviewer_ids:
        reviewer_model_dir = model_base / str(rid)
        if not reviewer_model_dir.exists():
            continue
        if rid not in experiment_data:
            continue
        texts, ratings, movie_ids = experiment_data[rid]
        labels = make_labels(ratings)

        for n_dir in sorted(reviewer_model_dir.iterdir(),
                            key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if not n_dir.name.isdigit():
                continue
            top_n = int(n_dir.name)
            if (rid, top_n) in done_keys:
                print(f"  [SKIP] reviewer {rid} N={top_n}: 既存結果あり")
                continue
            try:
                preds = predict_bert(texts, n_dir)
                m     = compute_metrics(labels, preds)
                df_summary = pd.concat([df_summary, pd.DataFrame([
                    {"reviewer_id": rid, "top_n": top_n, **m}
                ])], ignore_index=True)
                detail_df = pd.DataFrame([
                    {"reviewer_id": rid, "top_n": top_n, "movie_id": mid,
                     "review": t, "rating": r, "true_label": tl, "pred_label": pl}
                    for t, mid, r, tl, pl in zip(texts, movie_ids, ratings, labels, preds)
                ])
                _save_df(df_summary, out_summary)
                _append_detail_csv(detail_df, out_detail)
                print(f"  reviewer {rid} N={top_n}: {m}")
            except Exception as e:
                print(f"  [WARN] reviewer {rid} N={top_n}: {e}")


def evaluate_svmmodels(min_n: int, reviewer_ids: list[int],
                       experiment_data: dict[int, tuple],
                       out_summary: Path, out_detail: Path) -> None:
    model_base = MODELS_DIR / str(min_n) / "svmmodels"
    if not model_base.exists():
        return

    df_summary = _load_existing_df(out_summary)
    done_ids   = set(df_summary["reviewer_id"].tolist()) if not df_summary.empty else set()

    for rid in reviewer_ids:
        if rid in done_ids:
            print(f"  [SKIP] reviewer {rid}: 既存結果あり")
            continue
        model_dir = model_base / str(rid)
        if not model_dir.exists():
            continue
        if rid not in experiment_data:
            continue
        try:
            texts, ratings, movie_ids = experiment_data[rid]
            labels = make_labels(ratings)
            preds  = predict_svm(texts, movie_ids, rid, model_dir)
            m      = compute_metrics(labels, preds)
            df_summary = pd.concat([df_summary, pd.DataFrame([{"reviewer_id": rid, **m}])],
                                   ignore_index=True)
            detail_df = pd.DataFrame([
                {"reviewer_id": rid, "movie_id": mid, "review": t,
                 "rating": r, "true_label": tl, "pred_label": pl}
                for t, mid, r, tl, pl in zip(texts, movie_ids, ratings, labels, preds)
            ])
            _save_df(df_summary, out_summary)
            _append_detail_csv(detail_df, out_detail)
            print(f"  reviewer {rid}: {m}")
        except Exception as e:
            print(f"  [WARN] reviewer {rid}: {e}")


def evaluate_pctmodels(min_n: int, reviewer_ids: list[int],
                       experiment_data: dict[int, tuple],
                       out_summary: Path, out_detail: Path) -> None:
    model_base = MODELS_DIR / str(min_n) / "pctmodels"
    if not model_base.exists():
        return

    df_summary = _load_existing_df(out_summary)
    done_keys  = (set(zip(df_summary["reviewer_id"], df_summary["pct"]))
                  if not df_summary.empty else set())

    for rid in reviewer_ids:
        reviewer_model_dir = model_base / str(rid)
        if not reviewer_model_dir.exists():
            continue
        if rid not in experiment_data:
            continue
        texts, ratings, movie_ids = experiment_data[rid]
        labels = make_labels(ratings)

        pct_dirs = sorted(
            [p for p in reviewer_model_dir.iterdir()
             if p.is_dir() and p.name.startswith("pct") and p.name[3:].isdigit()],
            key=lambda p: int(p.name[3:]),
        )
        for pct_dir in pct_dirs:
            pct = int(pct_dir.name[3:])
            if (rid, pct) in done_keys:
                print(f"  [SKIP] reviewer {rid} pct={pct}%: 既存結果あり")
                continue
            try:
                preds = predict_bert(texts, pct_dir)
                m     = compute_metrics(labels, preds)
                df_summary = pd.concat([df_summary, pd.DataFrame([
                    {"reviewer_id": rid, "pct": pct, **m}
                ])], ignore_index=True)
                detail_df = pd.DataFrame([
                    {"reviewer_id": rid, "pct": pct, "movie_id": mid,
                     "review": t, "rating": r, "true_label": tl, "pred_label": pl}
                    for t, mid, r, tl, pl in zip(texts, movie_ids, ratings, labels, preds)
                ])
                _save_df(df_summary, out_summary)
                _append_detail_csv(detail_df, out_detail)
                print(f"  reviewer {rid} pct={pct}%: {m}")
            except Exception as e:
                print(f"  [WARN] reviewer {rid} pct={pct}%: {e}")


# ── ランダム分類 ──────────────────────────────────────────────────────────────

_N_RANDOM_VOTES = 11   # 多数決の試行回数（奇数推奨）

def evaluate_randommodel(min_n: int, reviewer_ids: list[int],
                         experiment_data: dict[int, tuple],
                         out_summary: Path, out_detail: Path) -> None:
    """
    各レビューを _N_RANDOM_VOTES 回ランダムに 0/1 分類し、
    過半数（>= ceil(N/2)）の結果を最終ラベルとして精度を計算する。
    モデルのロード不要・min_n はファイル名に使うだけ。
    """
    threshold = _N_RANDOM_VOTES // 2 + 1   # 11回なら 6

    df_summary = _load_existing_df(out_summary)
    done_ids   = set(df_summary["reviewer_id"].tolist()) if not df_summary.empty else set()

    for rid in reviewer_ids:
        if rid in done_ids:
            print(f"  [SKIP] reviewer {rid}: 既存結果あり")
            continue
        if rid not in experiment_data:
            continue
        try:
            texts, ratings, movie_ids = experiment_data[rid]
            labels = make_labels(ratings)
            preds = [
                1 if sum(random.randint(0, 1) for _ in range(_N_RANDOM_VOTES)) >= threshold else 0
                for _ in texts
            ]
            m = compute_metrics(labels, preds)
            df_summary = pd.concat(
                [df_summary, pd.DataFrame([{"reviewer_id": rid, **m}])],
                ignore_index=True,
            )
            detail_df = pd.DataFrame([
                {"reviewer_id": rid, "movie_id": mid, "review": t,
                 "rating": r, "true_label": tl, "pred_label": pl}
                for t, mid, r, tl, pl in zip(texts, movie_ids, ratings, labels, preds)
            ])
            _save_df(df_summary, out_summary)
            _append_detail_csv(detail_df, out_detail)
            print(f"  reviewer {rid}: {m}")
        except Exception as e:
            print(f"  [WARN] reviewer {rid}: {e}")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── モード選択 ──
    print("=" * 60)
    print("分類モードを選択してください")
    print("  1: SVM モデルで分類            (svmmodels)")
    print("  2: 全レビュー BERT で分類      (allmodels)")
    print("  3: TF-IDF 上位N件 BERT で分類  (nounmodels)")
    print("  4: TF-IDF 上位N%  BERT で分類  (pctmodels)")
    print(f"  5: ランダム分類（{_N_RANDOM_VOTES}回多数決）  (randommodel)")
    print("=" * 60)
    mode_input = input("モード (1/2/3/4/5): ").strip()
    while mode_input not in ("1", "2", "3", "4", "5"):
        mode_input = input("1, 2, 3, 4, または 5 を入力してください: ").strip()
    mode = int(mode_input)

    mode_label = {1: "svmmodels", 2: "allmodels", 3: "nounmodels",
                  4: "pctmodels",  5: "randommodel"}[mode]
    evaluate_fn = {
        1: evaluate_svmmodels,
        2: evaluate_allmodels,
        3: evaluate_nounmodels,
        4: evaluate_pctmodels,
        5: evaluate_randommodel,
    }[mode]

    min_counts   = find_min_movie_counts()
    reviewer_ids = reviewer_ids_in_experiment()

    # モード5はモデルディレクトリが不要なので min_counts が空でも動作させる
    if not min_counts:
        if mode != 5:
            print("[ERROR] modelsディレクトリ内にmin_movie_countディレクトリが見つかりません。")
            return
        min_counts = [0]   # ダミー（ファイル名に使用）
    print(f"\n[INFO] モード: {mode_label}")
    print(f"[INFO] min_movie_count: {min_counts}")
    print(f"[INFO] 対象レビュワー数: {len(reviewer_ids)}")

    # 実験データは min_movie_count に依存しないため、ここで一度だけ読み込む
    print("[INFO] レビュワーごとの実験データを読み込み中...")
    experiment_data: dict[int, tuple] = {}
    for rid in reviewer_ids:
        texts, ratings, movie_ids = load_experiment(rid)
        if texts:
            experiment_data[rid] = (texts, ratings, movie_ids)

    for min_n in min_counts:
        out_summary = OUTPUT_DIR / f"results_{mode_label}_{min_n}.xlsx"
        out_detail  = OUTPUT_DIR / f"detail_{mode_label}_{min_n}.csv"
        print(f"\n{'='*60}")
        print(f"[min_movie_count = {min_n}]  [{mode_label}]")
        evaluate_fn(min_n, reviewer_ids, experiment_data, out_summary, out_detail)

    print("\n[DONE]")


if __name__ == "__main__":
    main()