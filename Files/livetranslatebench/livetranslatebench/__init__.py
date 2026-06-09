"""LiveTranslateBench: a reproducible benchmark for real-time speech-to-speech translation.

Measures end-to-end lag, ASR transcript WER, and translation quality across
language pairs and noise levels, with a fair shared harness so every system is
fed identical audio at identical real-time pacing.
"""

__version__ = "0.1.0"
