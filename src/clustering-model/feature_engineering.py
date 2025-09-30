# ...existing code...
from pathlib import Path
import re
import json
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DEFAULT_JSON = Path(r"C:\Users\mathe\Downloads\dados\perfil_data_export.json")


def load_json_to_df(path: Path | str = DEFAULT_JSON) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_json(path, orient="records", lines=False)
    except ValueError:
        try:
            return pd.read_json(path, orient="records", lines=True)
        except ValueError:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.json_normalize(data)


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)            # remove HTML
    s = re.sub(r"\s+", " ", s).strip()        # normalize spaces
    s = s.lower()
    s = re.sub(r"[^\w\sáàâãéèêíïóôõöúçñªº-]", " ", s)  # remove punctuation but keep accents
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(clean_text)


def compute_tfidf(df: pd.DataFrame, text_col: str, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(df[text_col].astype(str).values)
    cols = [f"tfidf_{c}" for c in vec.get_feature_names_out()]
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(X, index=df.index, columns=cols)
    return tfidf_df, vec


def compute_counts(df: pd.DataFrame, text_col: str, max_features: int = 500, ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[pd.DataFrame, CountVectorizer]:
    vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(df[text_col].astype(str).values)
    cols = [f"cnt_{c}" for c in vec.get_feature_names_out()]
    cnt_df = pd.DataFrame.sparse.from_spmatrix(X, index=df.index, columns=cols)
    return cnt_df, vec


def try_embed(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def save_json_records(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient="records")
    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)


def select_text_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ("descricao", "descricao_vaga", "perfil", "texto", "title", "titulo", "summary", "job"))]
    if candidates:
        return candidates[0]
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        return obj_cols[0]
    raise RuntimeError("Nenhuma coluna de texto encontrada.")


def build_features(path: Path | str = DEFAULT_JSON, out_path: Path | None = None):
    df = load_json_to_df(path)
    text_col = select_text_column(df)
    df["_text_clean"] = preprocess_series(df[text_col])

    tfidf_df, _ = compute_tfidf(df, "_text_clean", max_features=1000, ngram_range=(1, 2))
    cnt_df, _ = compute_counts(df, "_text_clean", max_features=500, ngram_range=(1, 2))

    emb = try_embed(df["_text_clean"].tolist())
    if emb is not None:
        emb_df = pd.DataFrame(emb, index=df.index, columns=[f"emb_{i}" for i in range(emb.shape[1])])
        df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    out_df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True), cnt_df.reset_index(drop=True)], axis=1)

    if out_path is None:
        out_path = Path(path).parent / "perfil_data_features.json"

    # salvar apenas colunas relevantes para evitar arquivo gigante
    save_cols = ["_text_clean"]
    if emb is not None:
        save_cols += [c for c in out_df.columns if c.startswith("emb_")]
    save_cols += list(tfidf_df.columns[:50]) + list(cnt_df.columns[:30])
    save_cols = [c for c in save_cols if c in out_df.columns]

    save_json_records(out_df[save_cols], out_path)
    print("Saved features to:", out_path)


if __name__ == "__main__":
    build_features()