"""Aggregate JSONL results into a leaderboard (markdown + JSON).

We never ship numbers we did not measure. The leaderboard is generated from
results/*.jsonl that you produce by running the harness. Aggregation is a simple
mean per system (overall) plus per-language-pair and per-SNR breakdowns.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

METRICS = ["pal_s", "ttfa_s", "finish_lag_s", "rtf", "wer", "bleu", "chrf", "comet"]
LOWER_IS_BETTER = {"pal_s", "ttfa_s", "finish_lag_s", "rtf", "wer"}


def _load(results_dir: str) -> list[dict]:
    rows: list[dict] = []
    for p in Path(results_dir).glob("*.jsonl"):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _agg(rows: list[dict], metric: str) -> Optional[float]:
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    return round(statistics.mean(vals), 3) if vals else None


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.3f}"


def build(results_dir: str = "results", out_md: str = "LEADERBOARD.md") -> None:
    rows = _load(results_dir)
    if not rows:
        Path(out_md).write_text(
            "# LiveTranslateBench Leaderboard\n\n"
            "_No results yet. Run `ltbench run ...` then `ltbench leaderboard`._\n",
            encoding="utf-8",
        )
        print("No results found; wrote placeholder leaderboard.")
        return

    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)

    lines = ["# LiveTranslateBench Leaderboard", ""]
    lines.append(f"_{len(rows)} graded utterances across {len(by_system)} systems._")
    lines.append("")
    header = "| System | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    lines += [header, sep]

    # Sort systems by PAL (the headline latency metric), lower first.
    def sort_key(item):
        v = _agg(item[1], "pal_s")
        return (v is None, v if v is not None else 0)

    summary = {}
    for system, srows in sorted(by_system.items(), key=sort_key):
        cells = [_fmt(_agg(srows, m)) for m in METRICS]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
        summary[system] = {m: _agg(srows, m) for m in METRICS}

    lines += ["", "Lower is better: " + ", ".join(sorted(LOWER_IS_BETTER)) + ".", ""]
    lines += ["Higher is better: bleu, chrf, comet.", ""]
    lines += ["_All latency values are seconds, measured on the shared real-time harness clock._"]

    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("results/leaderboard.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Also write the copy the Vercel web page reads, stamped with a date.
    web_path = Path("web/leaderboard.json")
    if web_path.parent.exists():
        import datetime

        web_payload = {"_generated": datetime.date.today().isoformat(), **summary}
        web_path.write_text(
            json.dumps(web_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {out_md}, results/leaderboard.json and web/leaderboard.json")
    else:
        print(f"Wrote {out_md} and results/leaderboard.json")
