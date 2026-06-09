"""Latency metrics for streaming speech-to-speech translation.

The mistake most write-ups make is reporting a single "latency" number. In
continuous translation there are several distinct quantities, and they trade off
against each other. We report four, all computed from the shared harness clock:

1. TTFA (time-to-first-audio): seconds from the first input chunk sent to the
   first translated audio chunk received. Measures startup responsiveness.

2. Finish lag: seconds from the last input chunk sent to the last translated
   audio chunk received. Measures how far behind the speaker the translation is
   still trailing when the speaker stops. This is what listeners actually feel.

3. RTF (real-time factor): total wall time to fully translate divided by source
   audio duration. For a real-time system fed at 1x, this should sit near
   1.0 + (finish_lag / duration). Values >> 1 mean the system cannot keep up.

4. PAL (Proportional Average Lag): the headline number. We align the output
   transcript to the input transcript by *progress fraction* -- when the system
   has emitted f% of its final output text, how long ago did the input reach f%
   of its final text? Averaging that gap over the utterance gives a single,
   system-agnostic lag in seconds. It is a speech-to-speech adaptation of the
   Average Lagging idea from simultaneous-MT evaluation, made measurable without
   gold word alignments. Assumptions are documented on `average_lag` below.

All inputs are the OutputEvent list and the input send timestamps recorded by
the harness, so the same code scores every system identically.
"""

from __future__ import annotations

from statistics import mean
from typing import Optional

from ..systems.base import StreamResult


def time_to_first_audio(r: StreamResult) -> Optional[float]:
    for e in r.events:
        if e.audio:
            return e.t_recv - r.t_first_input
    return None


def finish_lag(r: StreamResult) -> Optional[float]:
    last_audio_t = None
    for e in r.events:
        if e.audio:
            last_audio_t = e.t_recv
    if last_audio_t is None:
        return None
    return last_audio_t - r.t_last_input


def real_time_factor(r: StreamResult) -> Optional[float]:
    last_t = None
    for e in r.events:
        if e.audio or e.is_final:
            last_t = e.t_recv
    if last_t is None or r.audio_duration_s <= 0:
        return None
    total_wall = last_t - r.t_first_input
    return total_wall / r.audio_duration_s


def _cumulative_timeline(r: StreamResult, attr: str) -> list[tuple[float, int]]:
    """Build [(t_recv, cumulative_char_count)] treating each event's text as a
    delta appended to the running transcript. If your SDK emits cumulative
    snapshots instead of deltas, set deltas=False in the harness and adjust here.
    """
    timeline: list[tuple[float, int]] = []
    total = 0
    for e in r.events:
        text = getattr(e, attr)
        if not text:
            continue
        total += len(text)
        timeline.append((e.t_recv, total))
    return timeline


def _time_reached(timeline: list[tuple[float, int]], target_count: float) -> Optional[float]:
    for t, cum in timeline:
        if cum >= target_count:
            return t
    return None


def average_lag(r: StreamResult) -> Optional[float]:
    """Proportional Average Lag (PAL), in seconds.

    Assumptions:
      - Output progress is proportional to input progress (the i-th fraction of
        translated text corresponds to the same fraction of source text). This
        holds well in aggregate for faithful translation and avoids needing gold
        alignments, but it is an approximation -- report it as PAL, not as a
        word-level lag.
      - Character counts proxy for content progress. For CJK targets you may want
        token counts; swap len(text) for a tokenizer in _cumulative_timeline.
    """
    in_tl = _cumulative_timeline(r, "input_transcript")
    out_tl = _cumulative_timeline(r, "output_transcript")
    if not in_tl or not out_tl:
        return None
    in_total = in_tl[-1][1]
    out_total = out_tl[-1][1]
    if in_total == 0 or out_total == 0:
        return None

    lags: list[float] = []
    for t_out, cum_out in out_tl:
        frac = cum_out / out_total
        target_in_count = frac * in_total
        t_in = _time_reached(in_tl, target_in_count)
        if t_in is not None:
            lags.append(t_out - t_in)
    return mean(lags) if lags else None


def all_latency_metrics(r: StreamResult) -> dict[str, Optional[float]]:
    return {
        "ttfa_s": time_to_first_audio(r),
        "finish_lag_s": finish_lag(r),
        "rtf": real_time_factor(r),
        "pal_s": average_lag(r),
    }
