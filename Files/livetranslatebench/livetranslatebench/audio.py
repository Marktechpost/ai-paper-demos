"""Audio utilities: load, resample, chunk, pace at real time, inject noise.

Everything is normalized to the Live API input contract before it reaches any
system: raw 16-bit PCM, 16 kHz, mono, little-endian, in 100 ms chunks.

The real-time pacer is load-bearing for the whole benchmark. If you feed audio
faster than 1x, latency numbers become meaningless -- a system would look fast
simply because it was handed the whole utterance up front. The pacer enforces
that every chunk is released no earlier than its real wall-clock position.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Iterator

import numpy as np

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHUNK_MS = 100
BYTES_PER_SAMPLE = 2


def load_pcm16_mono_16k(path: str) -> bytes:
    """Load any wav/flac/etc. as mono 16 kHz PCM16 little-endian bytes."""
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix
    if sr != SAMPLE_RATE_IN:
        mono = resample_poly(mono, SAMPLE_RATE_IN, sr)
    return _float_to_pcm16(mono)


def _float_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def _pcm16_to_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32767.0


def duration_seconds(pcm16_16k: bytes) -> float:
    n_samples = len(pcm16_16k) // BYTES_PER_SAMPLE
    return n_samples / SAMPLE_RATE_IN


def inject_noise(pcm16_16k: bytes, noise_pcm16_16k: bytes, snr_db: float) -> bytes:
    """Mix noise into speech at a target SNR (dB). Tiles/truncates noise to fit."""
    sig = _pcm16_to_float(pcm16_16k)
    noise = _pcm16_to_float(noise_pcm16_16k)
    if len(noise) < len(sig):
        reps = int(np.ceil(len(sig) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(sig)]

    sig_power = np.mean(sig**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return _float_to_pcm16(sig + noise)


def chunk_pcm16(pcm16_16k: bytes, chunk_ms: int = CHUNK_MS) -> Iterator[bytes]:
    bytes_per_chunk = int(SAMPLE_RATE_IN * (chunk_ms / 1000.0)) * BYTES_PER_SAMPLE
    for i in range(0, len(pcm16_16k), bytes_per_chunk):
        yield pcm16_16k[i : i + bytes_per_chunk]


async def realtime_pacer(
    pcm16_16k: bytes, chunk_ms: int = CHUNK_MS
) -> AsyncIterator[tuple[float, bytes]]:
    """Yield (t_send, chunk) at true real-time pace. t_send is time.monotonic()."""
    period = chunk_ms / 1000.0
    start = time.monotonic()
    for idx, chunk in enumerate(chunk_pcm16(pcm16_16k, chunk_ms)):
        target = start + idx * period
        now = time.monotonic()
        if target > now:
            await asyncio.sleep(target - now)
        yield time.monotonic(), chunk
