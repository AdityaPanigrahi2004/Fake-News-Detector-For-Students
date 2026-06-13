from __future__ import annotations

import re
from dataclasses import dataclass

CLICKBAIT_PATTERNS = [
    "shocking", "you won't believe", "secret", "viral", "forward this", "share immediately",
    "breaking!!!", "guaranteed", "miracle", "exposed", "banned", "hidden truth"
]
SOURCE_CUES = ["according to", "official", "reported by", "press release", "research", "study", "data", "source"]

@dataclass
class HeuristicResult:
    label: str
    score: int
    red_flags: list[str]
    positive_signals: list[str]

def heuristic_assessment(text: str) -> HeuristicResult:
    body = (text or "").lower()
    red_flags: list[str] = []
    positives: list[str] = []

    clickbait_hits = [p for p in CLICKBAIT_PATTERNS if p in body]
    if clickbait_hits:
        red_flags.append("Clickbait or urgency words found: " + ", ".join(clickbait_hits[:4]))

    if len(re.findall(r"!", text or "")) >= 3:
        red_flags.append("Excessive exclamation marks may indicate emotional manipulation.")

    if not any(cue in body for cue in SOURCE_CUES):
        red_flags.append("No clear official source, evidence, or data cue was found.")
    else:
        positives.append("The content includes at least one source/evidence cue.")

    if len((text or "").split()) < 40:
        red_flags.append("The content is short, so the system has limited evidence to verify.")

    risk = min(100, 25 + 18 * len(red_flags) - 10 * len(positives))
    credibility = max(0, 100 - risk)
    if credibility >= 70:
        label = "Real"
    elif credibility >= 45:
        label = "Suspicious"
    else:
        label = "Potentially Fake"

    return HeuristicResult(label=label, score=credibility, red_flags=red_flags, positive_signals=positives)
