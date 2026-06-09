"""Classic cascade baseline: streaming ASR -> NMT -> TTS.

This is the "old way" the launch is being compared against. It is wrapped in the
same streaming interface so the shared harness clocks it identically to the
native speech-to-speech systems. The cascade's latency penalty (each stage waits
on the previous, and TTS only starts after a translation unit is ready) is
exactly what we want to surface -- so do NOT pre-buffer the whole utterance.

Reference wiring (swap in whatever you standardize on, then pin versions in the
leaderboard metadata):
  - ASR:  faster-whisper (CTranslate2) in streaming/chunked mode
  - NMT:  NLLB-200 or a strong open MT model, translating committed ASR segments
  - TTS:  a low-latency streaming TTS (e.g. a fast neural vocoder model)

The skeleton below shows the intended control flow and event stamping. The three
TODOs are the only places real models plug in; everything around them (segment
commit policy, timing, event emission) is already harness-fair.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

from .base import OutputEvent, TranslationSystem


class CascadeBaseline(TranslationSystem):
    name = "whisper-nmt-tts-cascade"

    def __init__(self, commit_window_s: float = 1.0):
        # How much audio to accumulate before committing an ASR segment to MT.
        # Smaller = lower lag, worse quality. Report this value in run metadata;
        # it is the single biggest lever on a cascade's lag/quality trade-off.
        self.commit_window_s = commit_window_s
        self._asr = None
        self._mt = None
        self._tts = None

    def _lazy_init(self) -> None:
        if self._asr is not None:
            return
        # TODO(asr): load faster-whisper model here, e.g.
        #   from faster_whisper import WhisperModel
        #   self._asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
        # TODO(mt): load NMT model (e.g. transformers pipeline for NLLB-200).
        # TODO(tts): load a streaming TTS model.
        raise NotImplementedError(
            "Wire faster-whisper + NMT + TTS in CascadeBaseline._lazy_init(). "
            "See data/README.md for the exact versions used in the published run."
        )

    async def stream(
        self,
        audio_chunks: AsyncIterator[tuple[float, bytes]],
        *,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[OutputEvent]:
        self._lazy_init()

        buffer = bytearray()
        bytes_per_window = int(16000 * 2 * self.commit_window_s)  # 16k * 2 bytes * s

        async for _t_send, chunk in audio_chunks:
            buffer.extend(chunk)
            while len(buffer) >= bytes_per_window:
                window, buffer = bytes(buffer[:bytes_per_window]), buffer[bytes_per_window:]

                # 1) ASR on the committed window -> source text segment
                src_text = self._transcribe(window, source_lang)  # noqa: F841
                yield OutputEvent(time.monotonic(), input_transcript=src_text)

                # 2) NMT source segment -> target text segment
                tgt_text = self._translate(src_text, source_lang, target_lang)
                yield OutputEvent(time.monotonic(), output_transcript=tgt_text)

                # 3) TTS target segment -> PCM16 24k chunk(s)
                for pcm in self._synthesize(tgt_text, target_lang):
                    yield OutputEvent(time.monotonic(), audio=pcm)

        # Flush whatever remains in the buffer at end of stream (same 3 stages).
        if buffer:
            src_text = self._transcribe(bytes(buffer), source_lang)
            yield OutputEvent(time.monotonic(), input_transcript=src_text)
            tgt_text = self._translate(src_text, source_lang, target_lang)
            yield OutputEvent(time.monotonic(), output_transcript=tgt_text)
            for pcm in self._synthesize(tgt_text, target_lang):
                yield OutputEvent(time.monotonic(), audio=pcm)
        yield OutputEvent(time.monotonic(), is_final=True)

    # --- model calls (implement these three) ---------------------------------
    def _transcribe(self, pcm16_16k: bytes, source_lang: str) -> str:
        raise NotImplementedError

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

    def _synthesize(self, text: str, target_lang: str):
        raise NotImplementedError  # yield PCM16 24k chunks
