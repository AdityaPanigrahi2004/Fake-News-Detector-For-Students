from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/fake_news_model.joblib")

def normalize_label(value) -> str:
    text = str(value).strip().lower()
    if text in {"1", "fake", "false", "potentially fake", "unreliable"}:
        return "fake"
    if text in {"0", "real", "true", "reliable"}:
        return "real"
    return "fake" if "fake" in text or "false" in text else "real"

def load_training_frames(uploaded_files: Iterable) -> pd.DataFrame:
    frames = []
    for file in uploaded_files:
        df = pd.read_csv(file)
        name = getattr(file, "name", "").lower()
        if "text" not in df.columns:
            possible = [c for c in df.columns if c.lower() in {"title", "content", "article", "news"}]
            if possible:
                df["text"] = df[possible].fillna("").astype(str).agg(" ".join, axis=1)
        if "title" in df.columns:
            df["text"] = df.get("title", "").fillna("").astype(str) + " " + df.get("text", "").fillna("").astype(str)
        if "label" not in df.columns:
            if "fake" in name:
                df["label"] = "fake"
            elif "true" in name or "real" in name:
                df["label"] = "real"
        if "text" in df.columns and "label" in df.columns:
            frames.append(df[["text", "label"]])
    if not frames:
        raise ValueError("No usable CSV columns found. Need text+label, or Fake.csv/True.csv style files.")
    data = pd.concat(frames, ignore_index=True).dropna()
    data["label"] = data["label"].map(normalize_label)
    data = data[data["text"].str.len() > 20]
    return data

def train_and_save(data: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    joblib.dump(pipeline, model_path)
    return {"accuracy": float(accuracy_score(y_test, preds)), "report": report, "rows": int(len(data))}

def predict_text(text: str, model_path: Path = MODEL_PATH) -> dict | None:
    if not model_path.exists():
        return None
    pipeline = joblib.load(model_path)
    label = pipeline.predict([text])[0]
    probability = max(pipeline.predict_proba([text])[0]) if hasattr(pipeline[-1], "predict_proba") else 0.0
    return {"label": label, "confidence": float(probability)}
