# Project Report: Fake News Detector for Students

## Problem Statement

Misinformation spreads quickly through online news and social media, making it hard for students to differentiate between reliable and fake information. There is a need for an AI solution that can analyze articles, assess credibility, and provide concise, trustworthy summaries to prevent the spread of false information.

## Proposed Solution

TruthGuard AI is a Streamlit-based web application that accepts news content from text, URLs, screenshots/images, and PDFs. The system extracts text, preprocesses it, analyzes it using Google Gemini 1.5, and provides a credibility report with classification, confidence score, summary, red flags, and verification guidance.

## Technology Stack

- Frontend: Streamlit
- Backend: Python service modules
- AI API: Google Gemini 1.5 API
- ML: TF-IDF Vectorizer + Logistic Regression
- Dataset: Kaggle fake news CSV datasets
- OCR: pytesseract for image text extraction
- PDF: PyPDF2 for document text extraction

## System Approach

1. User submits text, URL, image, or PDF.
2. The extraction module converts input into plain text.
3. The preprocessing module cleans the text.
4. Gemini 1.5 evaluates credibility and generates an explanation.
5. The optional ML model predicts fake/real based on a Kaggle-trained classifier.
6. The scoring module combines AI, ML, and heuristic signals.
7. Streamlit displays a student-friendly report and stores it in history.

## Algorithm

- Text cleaning and normalization
- TF-IDF feature extraction
- Logistic Regression classification
- Gemini prompt-based credibility analysis
- Heuristic red-flag scoring for clickbait, emotional language, missing source cues, and suspicious wording

## Expected Result

The application helps students quickly verify online news, understand why content may be unreliable, and improve digital literacy.

## Future Scope

- Multilingual support for Indian regional languages
- Browser extension
- Mobile app
- Social media trend monitoring
- Deepfake image/video detection
- Integration with fact-checking APIs
