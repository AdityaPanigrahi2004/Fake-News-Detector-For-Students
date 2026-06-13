import re
from typing import Iterable

STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with",
    "this", "that", "by", "as", "from", "it", "be", "are", "was", "were"
}

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,!?%:;()'\"\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]+", clean_text(text).lower())

def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 2]

def prepare_for_model(text: str) -> str:
    return " ".join(remove_stopwords(tokenize(text)))
