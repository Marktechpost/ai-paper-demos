"""Benchmark harness.

For each (system, language-pair, noise-level, utterance) it:
  1. loads source audio as PCM16 16k mono,
  2. optionally mixes in noise at a target SNR,
  3. paces it at true real time into the system,
  4. records every OutputEvent on the shared clock,
  5. scores latency + WER + quality,
  6. appends one JSON record to results/<run>.jsonl.

The same paced input and the same clock are used for every system, which is the
property that makes the leaderboard fair.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import audio
from .metrics import all_latency_metrics, all_quality_metrics, word_error_rate
from .systems import REGISTRY
from .systems.base import OutputEvent, StreamResult, TranslationSystem


async def _collect(
    system: TranslationSystem,
    pcm: bytes,
    source_lang: str,
    target_lang: str,
    snr_db: Optional[float],
) -> StreamResult:
    pacer = audio.realtime_pacer(pcm)

    # Wrap the pacer so we capture first/last input send times for latency math.
    first_t = {"v": None}
    last_t = {"v": None}

    async def timed_pacer():
        async for t_send, chunk in pacer:
            if first_t["v"] is None:
                first_t["v"] = t_send
            last_t["v"] = t_send
            yield t_send, chunk

    events: list[OutputEvent] = []
    async for ev in system.stream(timed_pacer(), source_lang=source_lang, target_lang=target_lang):
        events.append(ev)

    return StreamResult(
        system=system.name,
        source_lang=source_lang,
        target_lang=target_lang,
        snr_db=snr_db,
        audio_duration_s=audio.duration_seconds(pcm),
        t_first_input=first_t["v"] if first_t["v"] is not None else time.monotonic(),
        t_last_input=last_t["v"] if last_t["v"] is not None else time.monotonic(),
        events=events,
    )


def score(r: StreamResult, source_ref: str, target_ref: str) -> dict:
    rec = {
        "system": r.system,
        "source_lang": r.source_lang,
        "target_lang": r.target_lang,
        "snr_db": r.snr_db,
        "audio_duration_s": round(r.audio_duration_s, 3),
        "output_audio_bytes": r.output_audio_bytes,
    }
    rec.update({k: _round(v) for k, v in all_latency_metrics(r).items()})
    rec["wer"] = _round(word_error_rate(r.input_transcript_text, source_ref))
    q = all_quality_metrics(r.output_transcript_text, target_ref, source_ref, r.target_lang)
    rec.update({k: _round(v) for k, v in q.items()})
    return rec


def _round(v):
    return round(v, 4) if isinstance(v, float) else v


def run_dataset(
    system_name: str,
    dataset: list[dict],
    snr_levels: list[Optional[float]],
    out_path: str,
    noise_pcm: Optional[bytes] = None,
) -> None:
    """dataset items: {audio_path, source_lang, target_lang, source_text, target_text}."""
    if system_name not in REGISTRY:
        raise SystemExit(f"Unknown system '{system_name}'. Known: {list(REGISTRY)}")
    system = REGISTRY[system_name]()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        for item in dataset:
            pcm_clean = audio.load_pcm16_mono_16k(item["audio_path"])
            for snr in snr_levels:
                if snr is None:
                    pcm = pcm_clean
                else:
                    if noise_pcm is None:
                        raise SystemExit("snr level requested but no --noise audio provided")
                    pcm = audio.inject_noise(pcm_clean, noise_pcm, snr)

                result = asyncio.run(
                    _collect(system, pcm, item["source_lang"], item["target_lang"], snr)
                )
                rec = score(result, item["source_text"], item["target_text"])
                rec["audio_path"] = item["audio_path"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(
                    f"[{system_name}] {item['source_lang']}->{item['target_lang']} "
                    f"snr={snr} pal={rec.get('pal_s')} ttfa={rec.get('ttfa_s')} "
                    f"chrf={rec.get('chrf')}"
                )
