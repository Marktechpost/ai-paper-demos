# Datasets

LiveTranslateBench is dataset-agnostic. You point it at a **manifest** (JSONL),
one utterance per line, with these fields:

```json
{"audio_path": "data/audio/es_0001.wav", "source_lang": "es", "target_lang": "en", "source_text": "Hola, ¿cómo estás?", "target_text": "Hello, how are you?"}
```

- `audio_path` — any format soundfile can read; it is normalized to 16 kHz mono PCM16.
- `source_lang` / `target_lang` — BCP-47 codes from the supported-language table.
- `source_text` — reference transcript of the source audio (used for WER).
- `target_text` — reference translation in the target language (used for BLEU/chrF/COMET).

## Recommended public sources

- **FLEURS** — parallel speech + text across 100+ languages. Good for many pairs
  with both `source_text` and aligned references.
- **CoVoST 2** — speech-to-text translation with reference translations; strong
  for X→English and English→X.
- **Common Voice** — source audio + transcripts when you need more `source_text`.

Pin the exact dataset version/split you use in your published run so the
leaderboard is reproducible.

## Noise

For the SNR sweep, supply a noise `.wav` via `--noise`. Recommended:

- **MUSAN** or **DEMAND** for real ambient/babble noise.

The harness mixes noise into the speech at each requested SNR (dB) before
streaming, so every system hears the identical degraded audio.

## Reproducibility metadata to record per run

Model/library versions for every system (Gemini model string, Whisper/NMT/TTS
checkpoints, Seamless checkpoint), dataset version, noise corpus, and the
cascade `commit_window_s`. These belong in the article's methodology box.
