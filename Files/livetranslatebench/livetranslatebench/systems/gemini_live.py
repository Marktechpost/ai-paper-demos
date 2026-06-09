"""Gemini 3.5 Live Translate adapter.

Streaming loop follows the Gemini Live API translation docs:
  - model: gemini-3.5-live-translate-preview
  - response modality: AUDIO only (translate mode does not support text/video/tools)
  - input audio: raw 16-bit PCM, 16 kHz, mono, little-endian, 100 ms chunks
  - output audio: raw 16-bit PCM, 24 kHz, mono, little-endian
  - input/output transcription enabled so we can score WER and translation quality

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from .base import OutputEvent, TranslationSystem


class GeminiLiveTranslate(TranslationSystem):
    name = "gemini-3.5-live-translate"
    model = "gemini-3.5-live-translate-preview"

    def __init__(self, echo_target_language: bool = False):
        # echo_target_language=False -> stay silent when the speaker is already
        # in the target language. That is the correct setting for a fair lag
        # measurement on cross-lingual material.
        self.echo_target_language = echo_target_language

    async def stream(
        self,
        audio_chunks: AsyncIterator[tuple[float, bytes]],
        *,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[OutputEvent]:
        # Imported lazily so the rest of the harness runs without the SDK installed.
        from google import genai
        from google.genai import types

        client = genai.Client()
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=target_lang,  # source is auto-detected
                echo_target_language=self.echo_target_language,
            ),
        )

        async with client.aio.live.connect(model=self.model, config=config) as session:

            async def pump() -> None:
                async for _t_send, chunk in audio_chunks:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                # Tell the server the input stream is done so it flushes the tail.
                await session.send_realtime_input(audio_stream_end=True)

            send_task = asyncio.create_task(pump())
            try:
                async for response in session.receive():
                    t = time.monotonic()
                    sc = getattr(response, "server_content", None)
                    if not sc:
                        continue
                    if sc.input_transcription and sc.input_transcription.text:
                        yield OutputEvent(t, input_transcript=sc.input_transcription.text)
                    if sc.output_transcription and sc.output_transcription.text:
                        yield OutputEvent(t, output_transcript=sc.output_transcription.text)
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                yield OutputEvent(t, audio=part.inline_data.data)
                    if getattr(sc, "turn_complete", False):
                        yield OutputEvent(time.monotonic(), is_final=True)
            finally:
                await send_task
