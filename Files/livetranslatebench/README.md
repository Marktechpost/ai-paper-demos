# LiveTranslateBench

**A reproducible benchmark for real-time, speech-to-speech translation.**

Most coverage of streaming translation models reports a single hand-wavy
"latency" number and a BLEU score on text. That tells you almost nothing about
what a listener actually experiences in a live conversation. LiveTranslateBench
measures the things that matter — *how far behind the speaker the translation
runs*, how well the system hears under noise, and how good the translation is —
and it measures every system the same way, on the same audio, on the same clock.

It ships with adapters for three systems so you can compare the new native
speech-to-speech approach against the classic pipeline:

| System | Approach |
|---|---|
| **Gemini 3.5 Live Translate** (`gemini-3.5-live-translate-preview`) | Native continuous speech-to-speech |
| **Whisper → NMT → TTS cascade** | The "old way": streaming ASR, then MT, then TTS |
| **Meta Seamless (streaming)** | Native simultaneous speech-to-speech |

> **Integrity note.** This repo ships **no pre-filled numbers**. The leaderboard
> is generated only from runs you execute. The cascade and Seamless adapters are
> clearly-marked wiring points, not guessed-at integrations, so nothing here
> claims a measurement it did not make.

---

## Why latency is the headline

In continuous translation there is no single "latency." There are at least four
distinct quantities, and they trade off against one another. LiveTranslateBench
reports all four, every one computed from the **same shared real-time clock**:

- **TTFA — time to first audio.** Seconds from the first input chunk to the
  first translated audio chunk. Startup responsiveness.
- **Finish lag.** Seconds from the *last* input chunk to the *last* translated
  audio chunk. How far behind the speaker the translation is still trailing when
  the speaker stops — the thing listeners feel most.
- **RTF — real-time factor.** Total wall time to fully translate ÷ source audio
  duration. Fed at 1×, a healthy real-time system sits near
  `1.0 + finish_lag/duration`; values well above that mean it cannot keep up.
- **PAL — Proportional Average Lag** (the headline). We align the output
  transcript to the input transcript by *progress fraction*: when the system has
  emitted f% of its final output, how long ago did the input reach f% of its
  final text? Averaged over the utterance, that gap is a single, system-agnostic
  lag in seconds. It is a speech-to-speech adaptation of the Average Lagging idea
  from simultaneous-MT evaluation, made measurable without gold word alignments.

### The one rule that makes the numbers real

Audio is streamed at **true real-time pace** — 100 ms chunks released no earlier
than their real wall-clock position. Feed audio faster than 1× and latency
numbers become meaningless: a system looks "fast" only because it was handed the
whole utterance up front. The pacer in `audio.py` enforces this for every system.

---

## Quality and accuracy metrics

- **WER** on the source-language input transcript — scores the *hearing* side
  (the ASR), against the dataset's source reference. Lower is better.
- **BLEU** and **chrF** (via `sacrebleu`) on the target-language output
  transcript, against the reference translation. chrF is reported alongside BLEU
  because it is more reliable for morphologically rich and non-space-delimited
  languages.
- **COMET** (optional, learned, correlates best with humans). Omitted gracefully
  if `unbabel-comet` is not installed, rather than failing the run.

Quality is scored on the **output transcript text**, not a re-transcription of
the output audio, so a second ASR pass cannot pollute the score.

---

## How the harness stays fair

```
                 ┌──────────────┐   identical paced audio   ┌────────────────────┐
 source audio ──▶│ real-time    │──────────────────────────▶│ system adapter      │
 (16k PCM16)     │ pacer (1×)   │   same monotonic clock     │ (Gemini / cascade / │
                 └──────────────┘                            │  Seamless)          │
                                                             └─────────┬──────────┘
                                                                       │ OutputEvents
                                                                       │ (audio + in/out transcripts,
                                                                       │  each stamped at receipt)
                                                                       ▼
                                              latency • WER • BLEU/chrF/COMET ──▶ JSONL ──▶ leaderboard
```

Every system implements one interface (`systems/base.py`): consume a paced
stream of `(t_send, pcm_chunk)` and yield `OutputEvent`s stamped with
`time.monotonic()` at the moment each result arrives. Same input, same clock,
same scoring code — that is the whole fairness argument.

---

## Install

```bash
git clone https://github.com/Marktechpost/LiveTranslateBench.git
cd LiveTranslateBench
pip install -e .
cp .env.example .env   # add your GEMINI_API_KEY
```

Optional extras:

```bash
pip install -e ".[comet]"     # learned quality metric
pip install -e ".[cascade]"   # faster-whisper + transformers for the baseline
```

## Quickstart

1. Build a manifest (one utterance per line). See [`data/README.md`](data/README.md):

```json
{"audio_path": "data/audio/es_0001.wav", "source_lang": "es", "target_lang": "en", "source_text": "Hola, ¿cómo estás?", "target_text": "Hello, how are you?"}
```

2. Run a system across a noise sweep:

```bash
ltbench run \
  --system gemini-3.5-live-translate \
  --manifest data/fleurs_es_en.jsonl \
  --snr clean,20,10,5 \
  --noise data/noise/babble_16k.wav \
  --out results/gemini.jsonl
```

3. Build the leaderboard:

```bash
ltbench leaderboard --results results --out LEADERBOARD.md
```

The leaderboard sorts by PAL (lowest lag first) and breaks out every metric per
system. `results/leaderboard.json` is emitted for programmatic use.

---

## What to test (and the honest hard cases)

The default language set in [`config/languages.yaml`](config/languages.yaml)
deliberately mixes easy and hard:

- High-resource, distant scripts (es↔en, en↔ja, zh↔en, ar↔en).
- **Language-detection stressors** — similar languages the docs flag as hard,
  e.g. Portuguese vs Spanish, Indonesian vs Malay.
- **Lower-resource targets** — Swahili, Tamil, Hindi — where quality usually
  separates systems most.
- A **noise sweep** (clean → 20 → 10 → 5 dB SNR), because robustness to loud,
  unpredictable environments is a headline claim worth verifying.

---

## Known limitations (report these, don't hide them)

- **Voice replication** in native speech-to-speech can drift across long pauses
  or rapid multi-speaker turns; this benchmark scores text + timing, not voice
  fidelity. A speaker-consistency metric is on the roadmap.
- **PAL** assumes output progress tracks input progress proportionally. It is an
  approximation reported as PAL, not a gold word-level lag.
- **Transcript deltas vs snapshots.** The harness treats transcript events as
  deltas. If your SDK emits cumulative snapshots, flip the handling in
  `metrics/latency.py::_cumulative_timeline`.
- Output audio is watermarked with SynthID by the provider; the benchmark does
  not rely on or evaluate that watermark.

## Roadmap

- Speaker-consistency / voice-drift metric for multi-speaker clips.
- Optional re-transcription-based quality metric (clearly separated from
  text-based scores).
- Per-pair and per-SNR plots in the leaderboard.
- A small, license-clean public sample set for one-command reproduction.

## License

Apache 2.0. Model outputs and datasets are subject to their own licenses.
