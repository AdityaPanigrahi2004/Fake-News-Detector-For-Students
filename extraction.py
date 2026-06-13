from __future__ import annotations

import io
import requests
from bs4 import BeautifulSoup
from PIL import Image
from PyPDF2 import PdfReader

def extract_from_url(url: str) -> str:
    response = requests.get(url, timeout=12, headers={"User-Agent": "TruthGuardAI/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["h1", "h2", "p"])]
    return "\n".join([p for p in paragraphs if len(p) > 30])[:12000]

def extract_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()

def extract_from_image(uploaded_file) -> str:
    try:
        import pytesseract
    except Exception as exc:
        raise RuntimeError("pytesseract is not installed or unavailable. Install Tesseract OCR for image extraction.") from exc
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image).strip()
