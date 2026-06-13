# TruthGuard AI - Fake News Detector for Students

TruthGuard AI is a Python + Streamlit capstone project for detecting fake or misleading news. It uses Google Gemini 1.5 for AI-based credibility analysis and also includes a Kaggle dataset training workflow using TF-IDF + Logistic Regression.

## Main Features

- Analyze pasted news text, article URLs, images/screenshots, and PDF files.
- Use Google Gemini 1.5 API to classify news as Real, Suspicious, or Potentially Fake.
- Generate student-friendly summaries, credibility scores, reasons, red flags, and verification tips.
- Train a machine learning model using Kaggle fake news datasets.
- Save analysis history locally for final demonstration.
- Run even without an API key using the built-in heuristic fallback.

## Recommended Kaggle Datasets

Use any one of these common Kaggle fake news datasets:

- Fake and Real News Dataset: files normally named `Fake.csv` and `True.csv`
- Fake News Detection Dataset: columns such as `title`, `text`, and `label`
- LIAR-style datasets can also be adapted if you rename columns to `text` and `label`

The Train Dataset page accepts one or more CSV files. If a filename contains `fake`, the app marks it as fake. If a filename contains `true` or `real`, the app marks it as real.

## Setup

1. Create a virtual environment.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Gemini key.

   Create `.env` from `.env.example` and add:

   ```bash
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-1.5-flash
   ```

4. Run the app.

   ```bash
   streamlit run app.py
   ```

## Project Structure

```text
truthguard_ai_fake_news_detector/
  app.py
  requirements.txt
  .env.example
  README.md
  project_report.md
  data/sample_news.csv
  src/
    gemini_service.py
    ml_pipeline.py
    preprocessing.py
    extraction.py
    scoring.py
```

## Deployment Notes

For Streamlit Cloud, push this folder to GitHub, create a new Streamlit app, and add `GEMINI_API_KEY` under app secrets. Do not commit the real API key.
