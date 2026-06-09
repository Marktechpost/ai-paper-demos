"""Meta Seamless streaming adapter (SeamlessStreaming / Seamless family).

Seamless exposes simultaneous speech-to-speech translation, which maps cleanly
onto our streaming interface. Because the exact public API surface and version
you standardize on matters for reproducibility, this adapter is intentionally a
thin, clearly-marked wiring point rather than a guessed-at integration. Pin the
checkpoint and library version in run metadata so the leaderboard is honest.

Intended approach:
  - Use the Seamless streaming inference agent (it accepts streaming audio and
    emits target audio + text incrementally).
  - Feed it the same paced 16 kHz PCM the harness feeds everyone else.
  - Stamp each emitted unit with time.monotonic() at emission.
  - Resample its output to PCM16 24k if needed so audio metrics align.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

from .base import OutputEvent, TranslationSystem


class SeamlessStreaming(TranslationSystem):
    name = "meta-seamless-streaming"

    def __init__(self, checkpoint: str = "seamless_streaming_unity"):
        self.checkpoint = checkpoint
        self._agent = None

    def _lazy_init(self) -> None:
        if self._agent is not None:
            return
        # TODO(seamless): construct the Seamless streaming inference agent here
        # and pin its version in results metadata. Keep the chunk-feeding loop in
        # stream() so latency is measured the same way as every other system.
        raise NotImplementedError(
            "Wire the Seamless streaming agent in SeamlessStreaming._lazy_init(). "
            "Pin the checkpoint + library version in data/README.md."
        )

    async def stream(
        self,
        audio_chunks: AsyncIterator[tuple[float, bytes]],
        *,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[OutputEvent]:
        self._lazy_init()
        async for _t_send, chunk in audio_chunks:
            # TODO(seamless): push `chunk` into the agent; for each emitted unit:
            #   yield OutputEvent(time.monotonic(), output_transcript=..., audio=...)
            _ = chunk
            raise NotImplementedError
        yield OutputEvent(time.monotonic(), is_final=True)
