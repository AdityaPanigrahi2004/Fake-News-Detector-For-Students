from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.extraction import extract_from_image, extract_from_pdf, extract_from_url
from src.gemini_service import analyze_with_gemini
from src.ml_pipeline import load_training_frames, predict_text, train_and_save
from src.preprocessing import clean_text
from src.scoring import heuristic_assessment

st.set_page_config(page_title="TruthGuard AI", page_icon="TG", layout="wide")

HISTORY_PATH = Path("data/history.csv")

def get_secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(name)

def save_history(result: dict, text: str) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "score", "summary", "sample"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": result.get("label", "Unknown"),
            "score": result.get("credibility_score", result.get("score", "")),
            "summary": result.get("summary", ""),
            "sample": text[:180]
        })

def normalize_result(gemini_result: dict | None, heuristic, ml_result: dict | None) -> dict:
    if gemini_result:
        result = gemini_result
    else:
        result = {
            "label": heuristic.label,
            "credibility_score": heuristic.score,
            "summary": "Gemini API was not available, so this report uses local heuristic checks.",
            "reasons": heuristic.positive_signals or ["Local text quality and source cues were checked."],
            "red_flags": heuristic.red_flags,
            "verification_steps": [
                "Search the headline on trusted news websites.",
                "Check the date, author, source, and original context.",
                "Compare the claim with official announcements or fact-checking websites."
            ],
            "trustworthy_rewrite": "Treat this claim as unverified until a reliable source confirms it.",
            "provider": "Local fallback"
        }
    if ml_result:
        result["ml_prediction"] = f"{ml_result['label']} ({ml_result['confidence']:.0%} confidence)"
    else:
        result["ml_prediction"] = "No trained Kaggle model found yet"
    result["heuristic_score"] = heuristic.score
    return result

st.title("TruthGuard AI - Fake News Detector for Students")
st.caption("Python + Streamlit frontend, Google Gemini 1.5 backend, and Kaggle dataset training support")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API key", value=get_secret("GEMINI_API_KEY") or "", type="password")
    model_name = st.text_input("Gemini model", value=get_secret("GEMINI_MODEL") or "gemini-1.5-flash")
    st.info("Do not hard-code your real API key in GitHub. Use .env locally or Streamlit secrets online.")

tab_analyze, tab_train, tab_history, tab_about = st.tabs(["Analyze News", "Train Kaggle Model", "History", "About Project"])

with tab_analyze:
    input_type = st.radio("Choose input type", ["Text", "URL", "Image", "PDF"], horizontal=True)
    article_text = ""

    if input_type == "Text":
        article_text = st.text_area("Paste news article or social media claim", height=240)
    elif input_type == "URL":
        url = st.text_input("Paste article URL")
        if url and st.button("Extract URL text"):
            try:
                article_text = extract_from_url(url)
                st.session_state["extracted_text"] = article_text
            except Exception as exc:
                st.error(f"Could not extract URL: {exc}")
        article_text = st.text_area("Extracted text", value=st.session_state.get("extracted_text", ""), height=220)
    elif input_type == "Image":
        image = st.file_uploader("Upload screenshot or news image", type=["png", "jpg", "jpeg"])
        if image and st.button("Extract image text"):
            try:
                st.session_state["image_text"] = extract_from_image(image)
            except Exception as exc:
                st.error(str(exc))
        article_text = st.text_area("Extracted image text", value=st.session_state.get("image_text", ""), height=220)
    else:
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf and st.button("Extract PDF text"):
            try:
                st.session_state["pdf_text"] = extract_from_pdf(pdf)
            except Exception as exc:
                st.error(f"Could not read PDF: {exc}")
        article_text = st.text_area("Extracted PDF text", value=st.session_state.get("pdf_text", ""), height=220)

    if st.button("Analyze Credibility", type="primary"):
        cleaned = clean_text(article_text)
        if len(cleaned.split()) < 12:
            st.warning("Please enter more news content for better analysis.")
        else:
            heuristic = heuristic_assessment(cleaned)
            ml_result = predict_text(cleaned)
            gemini_result = None
            try:
                gemini_result = analyze_with_gemini(cleaned, api_key=api_key or None, model_name=model_name)
            except Exception as exc:
                st.warning(f"Gemini analysis unavailable: {exc}")
            result = normalize_result(gemini_result, heuristic, ml_result)
            save_history(result, cleaned)

            c1, c2, c3 = st.columns(3)
            c1.metric("AI Classification", result.get("label", "Unknown"))
            c2.metric("Credibility Score", result.get("credibility_score", "NA"))
            c3.metric("ML Model", result.get("ml_prediction", "Not trained"))

            st.subheader("Student-Friendly Summary")
            st.write(result.get("summary", "No summary available."))

            left, right = st.columns(2)
            with left:
                st.subheader("Reasons")
                for item in result.get("reasons", []):
                    st.write(f"- {item}")
                st.subheader("Red Flags")
                flags = result.get("red_flags", []) or ["No major red flags found."]
                for item in flags:
                    st.write(f"- {item}")
            with right:
                st.subheader("Verification Steps")
                for item in result.get("verification_steps", []):
                    st.write(f"- {item}")
                st.subheader("Responsible Rewrite")
                st.write(result.get("trustworthy_rewrite", "Verify the claim before sharing."))

with tab_train:
    st.subheader("Train with Kaggle CSV files")
    st.write("Upload Fake.csv and True.csv, or a CSV containing text and label columns.")
    csv_files = st.file_uploader("Upload CSV datasets", type=["csv"], accept_multiple_files=True)
    if st.button("Train Model"):
        if not csv_files:
            st.warning("Upload at least one CSV file.")
        else:
            try:
                data = load_training_frames(csv_files)
                metrics = train_and_save(data)
                st.success(f"Model trained on {metrics['rows']} rows")
                st.metric("Validation Accuracy", f"{metrics['accuracy']:.2%}")
                st.json(metrics["report"])
            except Exception as exc:
                st.error(f"Training failed: {exc}")

with tab_history:
    st.subheader("Analysis History")
    if HISTORY_PATH.exists():
        st.dataframe(pd.read_csv(HISTORY_PATH), use_container_width=True)
    else:
        st.info("No history yet. Analyze an article first.")

with tab_about:
    st.subheader("Certification Project Summary")
    st.write("TruthGuard AI helps students verify news before sharing it. It combines Gemini 1.5 reasoning, optional Kaggle-trained ML classification, OCR/PDF/URL extraction, and a simple Streamlit interface.")
    st.write("Use this app for educational credibility support, not as a legal or absolute truth decision system.")
