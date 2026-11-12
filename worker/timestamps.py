"""Validate ASR word times and group words into utterances, without rewriting recognized text."""
from __future__ import annotations

import math
import re

SAMPLE_RATE = 16000
# Qwen3-ASR handles long audio and its forced aligner up to 5 minutes: a whole checkpoint window
# (AUDIO_CHUNK_SECONDS <= 120, plus overlap) fits in one pass, so speech keeps its full context.
MAX_SEGMENT_SECONDS = 130


def speech_segments(regions: list[dict], total_samples: int, max_seconds: int = MAX_SEGMENT_SECONDS) -> list[tuple[int, int]]:
    """Merge consecutive Silero regions (in samples) into spans no longer than max_seconds.

    Leading and trailing silence is never sent to the models; a silent window is skipped entirely.
    A single region longer than the limit is split at fixed boundaries.
    """
    limit = max_seconds * SAMPLE_RATE
    spans: list[tuple[int, int]] = []
    for region in sorted(regions, key=lambda item: item["start"]):
        start, end = max(0, int(region["start"])), min(total_samples, int(region["end"]))
        if end <= start:
            continue
        if spans and end - spans[-1][0] <= limit:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
            continue
        while end - start > limit:
            spans.append((start, start + limit))
            start += limit
        spans.append((start, end))
    return spans


# Qwen3-ASR can drop all punctuation and capitals for a segment that starts mid-sentence.
UNPUNCTUATED_MIN_WORDS = 20
PUNCTUATION = re.compile(r"[.,;:!?…。，！？]")


def is_unpunctuated(text: str) -> bool:
    return len(text.split()) >= UNPUNCTUATED_MIN_WORDS and not PUNCTUATION.search(text)


def first_pause(words: list[dict], length: float, within: float = 15.0) -> float | None:
    """Seconds (span-relative) of the widest gap between aligned words early in the span.

    Cutting there lets the rest of the span start on a new phrase; Silero often sees no pause at all
    in continuous speech over music, so the aligner's word times are used instead.
    """
    best, cut = 0.0, None
    for current, following in zip(words, words[1:]):
        if following["start"] > within:
            break
        gap = following["start"] - current["end"]
        middle = (current["end"] + following["start"]) / 2
        if gap > best and 1.0 <= middle <= length - 5.0:
            best, cut = gap, middle
    return cut


def align_words(text: str, aligned: list[dict], split) -> list[dict]:
    """Attach aligner times to the recognized words, keeping Qwen3-ASR's own spelling and punctuation.

    The aligner times the punctuation-free units produced by ``split`` (the processor's own splitter);
    each whitespace-delimited word maps to its consecutive units, so "Bonjour !" or "l’homme," keep
    their typography. A token without alignable characters (a lone "!" or "«") joins its neighbour.
    """
    words, position, pending = [], 0, ""
    for token in text.split():
        count = len(split(token))
        if not count:
            if words:
                words[-1]["text"] += " " + token
            else:
                pending += token + " "
            continue
        units = aligned[position:position + count]
        if len(units) != count:
            raise ValueError("ASR : alignement incomplet par rapport au texte reconnu.")
        position += count
        words.append({"text": pending + token, "start": units[0]["start_time"], "end": units[-1]["end_time"]})
        pending = ""
    if position != len(aligned):
        raise ValueError("ASR : alignement incohérent avec le texte reconnu.")
    if pending and words:
        words[-1]["text"] += " " + pending.strip()
    return words


def _seconds(value) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("ASR : horodatage de mot invalide.")
    return float(value)


def timed_words(result: dict, window: dict) -> list[dict]:
    """Convert window-relative word times (seconds) into absolute milliseconds."""
    words = result.get("words")
    if not isinstance(words, list):
        raise ValueError("ASR : mots horodatés absents.")
    length = window["endMs"] - window["startMs"]
    timed, previous_start = [], 0
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("text"), str):
            raise ValueError("ASR : mot invalide.")
        start, end = _seconds(word.get("start")), _seconds(word.get("end"))
        start_ms, end_ms = round(start * 1000), round(end * 1000)
        if start_ms < previous_start or end_ms < start_ms or end_ms > length + 250:
            raise ValueError("ASR : horodatage hors du morceau audio ou non ordonné.")
        previous_start = start_ms
        text = word["text"].strip()
        if text:
            timed.append({"text": text, "startMs": window["startMs"] + min(start_ms, length),
                          "endMs": window["startMs"] + min(end_ms, length)})
    return timed


def utterances(result: dict, window: dict) -> tuple[list[dict], list[dict]]:
    words = timed_words(result, window)
    selected = [word for word in words
                if window["coreStartMs"] <= (word["startMs"] + word["endMs"]) / 2 < window["coreEndMs"]]
    entries, group = [], []
    for index, word in enumerate(selected):
        if group and word["startMs"] - group[-1]["endMs"] > 800:
            entries.append(_entry(group, window))
            group = []
        group.append(word)
        if (index == len(selected) - 1 or re.search(r"[.!?…]$", word["text"])
                or word["endMs"] - group[0]["startMs"] >= 12000
                or sum(len(w["text"]) + 1 for w in group) >= 240):
            entries.append(_entry(group, window))
            group = []
    return entries, words


def _entry(words: list[dict], window: dict) -> dict:
    return {"speaker": "Indéterminé", "startMs": words[0]["startMs"], "endMs": words[-1]["endMs"],
            "text": " ".join(w["text"] for w in words), "chunkIndex": window["index"] + 1}


def audio_context(transcript: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    return [{"startMs": t["startMs"], "endMs": t["endMs"], "text": t["text"]}
            for t in transcript if t["startMs"] < end_ms and t["endMs"] >= start_ms]
