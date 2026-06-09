"""LiveTranslateBench CLI.

Examples:
  ltbench run --system gemini-3.5-live-translate \\
      --manifest data/fleurs_es_en.jsonl --snr clean,20,10,5 \\
      --noise data/noise/babble_16k.wav --out results/gemini.jsonl

  ltbench leaderboard --results results --out LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import leaderboard
from .audio import load_pcm16_mono_16k
from .harness import run_dataset


def _parse_snr(s: str):
    levels = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        levels.append(None if tok in {"clean", "inf", "none"} else float(tok))
    return levels


def _load_manifest(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    required = {"audio_path", "source_lang", "target_lang", "source_text", "target_text"}
    for it in items:
        missing = required - it.keys()
        if missing:
            raise SystemExit(f"Manifest item missing fields {missing}: {it}")
    return items


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ltbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a system over a manifest")
    pr.add_argument("--system", required=True)
    pr.add_argument("--manifest", required=True, help="JSONL manifest of utterances")
    pr.add_argument("--snr", default="clean", help="comma list, e.g. clean,20,10,5")
    pr.add_argument("--noise", default=None, help="noise wav for SNR mixing")
    pr.add_argument("--out", required=True)

    pl = sub.add_parser("leaderboard", help="aggregate results into a leaderboard")
    pl.add_argument("--results", default="results")
    pl.add_argument("--out", default="LEADERBOARD.md")

    args = p.parse_args(argv)

    if args.cmd == "run":
        dataset = _load_manifest(args.manifest)
        snr_levels = _parse_snr(args.snr)
        noise_pcm = load_pcm16_mono_16k(args.noise) if args.noise else None
        run_dataset(args.system, dataset, snr_levels, args.out, noise_pcm)
    elif args.cmd == "leaderboard":
        Path(args.results).mkdir(parents=True, exist_ok=True)
        leaderboard.build(args.results, args.out)


if __name__ == "__main__":
    main()
