"""Translation quality on the target-language output transcript.

We score the system's output transcript against the dataset's reference
translation. Two families:

  - Surface overlap: BLEU and chrF (via sacrebleu). Cheap, deterministic, always
    available. chrF is more reliable than BLEU for morphologically rich and
    non-space-delimited languages, so we report both.
  - Learned quality: COMET (optional, heavy). Correlates better with human
    judgement. Requires the `unbabel-comet` package and a model download; if it
    is not installed we simply omit the column rather than fail the run.

Scoring text rather than re-transcribing the output audio keeps the metric from
being polluted by a second ASR pass. Re-transcription-based scoring can be added
later as a separate, clearly-labelled metric.
"""

from __future__ import annotations

from typing import Optional

_comet_model = None


def bleu(hypothesis: str, reference: str, target_lang: str) -> Optional[float]:
    try:
        import sacrebleu
    except ImportError:
        return None
    # Use the spm-based tokenizer for CJK/Thai etc.; intl tokenizer otherwise.
    tok = "flores200" if target_lang in {"zh-Hans", "zh-Hant", "ja", "th", "ko"} else "13a"
    return sacrebleu.sentence_bleu(hypothesis, [reference], tokenize=tok).score


def chrf(hypothesis: str, reference: str) -> Optional[float]:
    try:
        import sacrebleu
    except ImportError:
        return None
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score


def comet(hypothesis: str, reference: str, source: str) -> Optional[float]:
    """COMET reference-based score. Returns None if COMET is unavailable."""
    global _comet_model
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        return None
    if _comet_model is None:
        path = download_model("Unbabel/wmt22-comet-da")
        _comet_model = load_from_checkpoint(path)
    data = [{"src": source, "mt": hypothesis, "ref": reference}]
    out = _comet_model.predict(data, batch_size=1, gpus=0, progress_bar=False)
    return float(out["system_score"])


def all_quality_metrics(
    hypothesis: str, reference: str, source: str, target_lang: str
) -> dict[str, Optional[float]]:
    return {
        "bleu": bleu(hypothesis, reference, target_lang),
        "chrf": chrf(hypothesis, reference),
        "comet": comet(hypothesis, reference, source),
    }
