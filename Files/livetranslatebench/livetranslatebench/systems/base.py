"""Common interface every translation system implements.

The harness paces source audio at real time and hands each system an async
stream of (t_send, pcm_chunk) tuples. Each system yields OutputEvent objects
tagged with the monotonic time the harness observed them. Because all systems
share the same clock and the same paced input, their latency numbers are
directly comparable -- that comparability is the point of this project.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class OutputEvent:
    """A single observation emitted by a system while translating."""

    t_recv: float  # time.monotonic() seconds, stamped by the harness clock
    audio: Optional[bytes] = None  # translated PCM16 chunk (24 kHz mono), if any
    input_transcript: Optional[str] = None  # incremental source-language text
    output_transcript: Optional[str] = None  # incremental target-language text
    is_final: bool = False


@dataclass
class StreamResult:
    """Everything observed during one utterance, plus run metadata."""

    system: str
    source_lang: str
    target_lang: str
    snr_db: Optional[float]  # None == clean
    audio_duration_s: float
    t_first_input: float
    t_last_input: float
    events: list[OutputEvent] = field(default_factory=list)

    @property
    def output_audio_bytes(self) -> int:
        return sum(len(e.audio) for e in self.events if e.audio)

    @property
    def input_transcript_text(self) -> str:
        return "".join(e.input_transcript for e in self.events if e.input_transcript)

    @property
    def output_transcript_text(self) -> str:
        return "".join(e.output_transcript for e in self.events if e.output_transcript)


class TranslationSystem(abc.ABC):
    """Base class for any system under test."""

    name: str = "abstract"

    @abc.abstractmethod
    async def stream(
        self,
        audio_chunks: AsyncIterator[tuple[float, bytes]],
        *,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[OutputEvent]:
        """Consume paced (t_send, pcm16_16k_chunk) tuples; yield OutputEvents.

        Implementations MUST stamp each yielded event with time.monotonic() at
        the moment the result becomes available -- not when input was sent.
        """
        raise NotImplementedError
        yield  # pragma: no cover  (makes this an async generator for typing)
