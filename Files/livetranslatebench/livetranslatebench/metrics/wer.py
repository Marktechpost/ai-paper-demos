"""Word Error Rate on the source-language input transcript.

This scores the ASR side of each system: how well it heard the speaker. We
compare the system's input transcript against the dataset's source reference
text. Lower is better. Normalization (lowercasing, punctuation stripping) is
applied to both sides so systems are not penalized for cosmetic differences.
"""

from __future__ import annotations

import re
from typing import Optional


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def word_error_rate(hypothesis: str, reference: str) -> Optional[float]:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return None
    try:
        import jiwer

        return jiwer.wer(ref, hyp)
    except ImportError:
        # Minimal Levenshtein fallback so the harness runs without jiwer.
        return _levenshtein_wer(hyp.split(), ref.split())


def _levenshtein_wer(hyp: list[str], ref: list[str]) -> float:
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + cost)
            prev = cur
    return d[m] / n
