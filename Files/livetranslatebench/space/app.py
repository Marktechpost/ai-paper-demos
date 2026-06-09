"""LiveTranslateBench — Hugging Face Space.

Two tabs:
  1. Try it     — upload a short clip, translate it with Gemini 3.5 Live Translate,
                  hear the result, and see the four latency metrics measured the
                  same way the benchmark measures them.
  2. Leaderboard — render results/leaderboard.json if you commit one to the Space.

Gemini runs on Google's side, so this Space needs no GPU. Set GEMINI_API_KEY as
a Space secret (Settings -> Variables and secrets). The cascade and Seamless
baselines are intentionally NOT served here; run those locally with the CLI.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import wave
from pathlib import Path

import gradio as gr

from livetranslatebench import audio
from livetranslatebench.metrics import all_latency_metrics
from livetranslatebench.systems.base import OutputEvent, StreamResult
from livetranslatebench.systems.gemini_live import GeminiLiveTranslate

MAX_SECONDS = 30  # keep Space runs short and responsive
OUTPUT_RATE = 24000

# A friendly subset of the 70+ supported BCP-47 targets; extend freely.
TARGETS = {
    "English (en)": "en",
    "Spanish (es)": "es",
    "French (fr)": "fr",
    "German (de)": "de",
    "Hindi (hi)": "hi",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Chinese, Simplified (zh-Hans)": "zh-Hans",
    "Arabic (ar)": "ar",
    "Portuguese, Brazil (pt-BR)": "pt-BR",
    "Swahili (sw)": "sw",
    "Tamil (ta)": "ta",
}


def _pcm24k_to_wav(pcm_bytes: bytes) -> str:
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUTPUT_RATE)
        w.writeframes(pcm_bytes)
    return path


async def _collect(pcm: bytes, target_lang: str) -> StreamResult:
    system = GeminiLiveTranslate()
    first_t = {"v": None}
    last_t = {"v": None}

    async def timed_pacer():
        async for t_send, chunk in audio.realtime_pacer(pcm):
            if first_t["v"] is None:
                first_t["v"] = t_send
            last_t["v"] = t_send
            yield t_send, chunk

    events: list[OutputEvent] = []
    async for ev in system.stream(timed_pacer(), source_lang="auto", target_lang=target_lang):
        events.append(ev)

    return StreamResult(
        system=system.name,
        source_lang="auto",
        target_lang=target_lang,
        snr_db=None,
        audio_duration_s=audio.duration_seconds(pcm),
        t_first_input=first_t["v"] or time.monotonic(),
        t_last_input=last_t["v"] or time.monotonic(),
        events=events,
    )


def translate(audio_path, target_label):
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise gr.Error("Set GEMINI_API_KEY as a Space secret (Settings -> Variables and secrets).")
    if not audio_path:
        raise gr.Error("Upload or record a short clip first.")

    target_lang = TARGETS[target_label]
    pcm = audio.load_pcm16_mono_16k(audio_path)
    pcm = pcm[: MAX_SECONDS * 16000 * 2]  # trim to keep runs short

    result = asyncio.run(_collect(pcm, target_lang))

    out_audio_bytes = b"".join(e.audio for e in result.events if e.audio)
    out_wav = _pcm24k_to_wav(out_audio_bytes) if out_audio_bytes else None

    m = all_latency_metrics(result)
    rows = [
        ["PAL — average lag (s)", _fmt(m["pal_s"])],
        ["TTFA — time to first audio (s)", _fmt(m["ttfa_s"])],
        ["Finish lag (s)", _fmt(m["finish_lag_s"])],
        ["Real-time factor", _fmt(m["rtf"])],
    ]
    return (
        out_wav,
        result.input_transcript_text or "(no transcript)",
        result.output_transcript_text or "(no transcript)",
        rows,
    )


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def load_leaderboard():
    for cand in ("results/leaderboard.json", "leaderboard.json"):
        p = Path(cand)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            cols = ["system", "pal_s", "ttfa_s", "finish_lag_s", "rtf", "wer", "bleu", "chrf", "comet"]
            rows = []
            for system, vals in data.items():
                rows.append([system] + [vals.get(c) for c in cols[1:]])
            return cols, rows
    return (
        ["system", "pal_s", "ttfa_s", "finish_lag_s", "rtf", "wer", "bleu", "chrf", "comet"],
        [["No results committed yet — run the CLI and commit results/leaderboard.json"] + [None] * 8],
    )


with gr.Blocks(title="LiveTranslateBench") as demo:
    gr.Markdown(
        "# 🎙️ LiveTranslateBench\n"
        "Real-time speech-to-speech translation, measured honestly. "
        "The **Try it** tab runs your clip through Gemini 3.5 Live Translate and reports the "
        "same four latency metrics the benchmark uses. Baselines (cascade, Seamless) run via the CLI."
    )

    with gr.Tab("Try it"):
        with gr.Row():
            with gr.Column():
                in_audio = gr.Audio(type="filepath", label=f"Source audio (≤ {MAX_SECONDS}s)")
                tgt = gr.Dropdown(choices=list(TARGETS), value="English (en)", label="Translate into")
                go = gr.Button("Translate", variant="primary")
            with gr.Column():
                out_audio = gr.Audio(label="Translated audio (24 kHz)")
                in_tx = gr.Textbox(label="Source transcript", lines=3)
                out_tx = gr.Textbox(label="Translated transcript", lines=3)
                metrics = gr.Dataframe(headers=["metric", "value"], label="Measured latency", wrap=True)
        go.click(translate, [in_audio, tgt], [out_audio, in_tx, out_tx, metrics])
        gr.Markdown(
            "_Runs are paced at true real time, so a 20s clip takes ~20s — that pacing is what "
            "makes the latency numbers meaningful. Output audio is watermarked with SynthID by the provider._"
        )

    with gr.Tab("Leaderboard"):
        cols, rows = load_leaderboard()
        gr.Markdown("Lower is better: pal_s, ttfa_s, finish_lag_s, rtf, wer. Higher is better: bleu, chrf, comet.")
        gr.Dataframe(headers=cols, value=rows, wrap=True, label="LiveTranslateBench")


if __name__ == "__main__":
    demo.launch()
