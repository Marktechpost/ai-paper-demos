"""System registry. Add new systems here so the CLI can find them by name."""

from .base import OutputEvent, StreamResult, TranslationSystem
from .gemini_live import GeminiLiveTranslate
from .cascade import CascadeBaseline
from .seamless import SeamlessStreaming

REGISTRY = {
    GeminiLiveTranslate.name: GeminiLiveTranslate,
    CascadeBaseline.name: CascadeBaseline,
    SeamlessStreaming.name: SeamlessStreaming,
}

__all__ = [
    "OutputEvent",
    "StreamResult",
    "TranslationSystem",
    "GeminiLiveTranslate",
    "CascadeBaseline",
    "SeamlessStreaming",
    "REGISTRY",
]
