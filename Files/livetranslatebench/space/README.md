---
title: LiveTranslateBench
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
short_description: Real-time speech-to-speech translation, measured honestly.
---

# LiveTranslateBench (Space)

Interactive demo + leaderboard for [LiveTranslateBench](https://github.com/Marktechpost/LiveTranslateBench),
a reproducible benchmark for real-time speech-to-speech translation.

- **Try it** — upload a short clip, translate it with Gemini 3.5 Live Translate,
  and see the four latency metrics (PAL, TTFA, finish lag, RTF) measured the same
  way the benchmark measures them.
- **Leaderboard** — renders `results/leaderboard.json` if committed to this Space.

## Setup

This Space needs one secret: `GEMINI_API_KEY`
(Settings → Variables and secrets). Gemini runs server-side at Google, so the
Space runs fine on free **CPU basic** hardware — no GPU required.

The heavy baselines (Whisper→NMT→TTS cascade, Meta Seamless) are **not** served
here; run them locally with the `ltbench` CLI and commit the resulting
`leaderboard.json` to populate the Leaderboard tab.
