#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 from the chart's recent reads.

Method: recent-read empirical occupancy. The chart churns fast, so the
forecast trusts what the newest reads actually showed instead of any
parametric model of how games move. Only the most recent reads count,
the newest ones count the most (a 0.85 recency ramp), and each place's
distribution is the weighted share of those reads in which each game
sat at that place, lightly smoothed toward the game's overall top-10
presence, then sharpened a little and trimmed: a small slice of every
place stays unnamed on purpose, because whatever mass a place leaves
unnamed is shared by the whole chart at scoring time and a thin spread
over long-shot games scores far worse than that shared slice.

Reads /input/history.ndjson (or $MAKEFUN_INPUT_DIR) and writes
forecast.json to /output (or $MAKEFUN_OUTPUT_DIR). Standard library
only, no network, fully deterministic: same history, same bytes.
"""

import json
import os

PLACES = 10
RECENT_READS = 14        # K: how many of the newest reads to trust
RECENCY_DECAY = 0.85     # each read older than the newest weighs 0.85x more
SMOOTHING = 0.6          # A: pull toward a game's overall top-10 presence
SHARPENING = 1.15        # exponent that rewards confident occupancy
RESERVE = 0.05           # share of every place deliberately left unnamed
MIN_PROB = 0.01          # rows below this are dropped into the reserve


def read_charts(input_dir):
    """Parse history.ndjson into charts, oldest first.

    Each chart becomes a dict {place: universe_id} holding places 1..10
    only. Lines that are not JSON, carry no chart, or name no game in
    the top 10 are skipped, so a damaged history degrades instead of
    crashing the run.
    """
    path = os.path.join(input_dir, "history.ndjson")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    charts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        entries = record.get("up_and_coming")
        if not isinstance(entries, list):
            continue
        chart = {}
        seen = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            uid = entry.get("universe_id")
            if not isinstance(uid, int) or uid in seen:
                continue
            rank = entry.get("rank")
            place = rank if isinstance(rank, int) else index + 1
            if 1 <= place <= PLACES and place not in chart:
                chart[place] = uid
                seen.add(uid)
        if chart:
            charts.append(chart)
    return charts


def occupancy_forecast(charts):
    """Build (rows, top10) from full top-10 charts by weighted occupancy."""
    recent = charts[-RECENT_READS:]
    count = len(recent)
    # Oldest of the window first; the newest read (last) weighs 1.0.
    weights = [RECENCY_DECAY ** (count - 1 - i) for i in range(count)]
    total_weight = sum(weights)

    place_hits = {}   # (place, uid) -> weighted reads with uid at place
    anywhere = {}     # uid -> weighted reads with uid anywhere in top 10
    rank_moment = {}  # uid -> sum of weight * place over its appearances
    for chart, weight in zip(recent, weights):
        for place, uid in chart.items():
            key = (place, uid)
            place_hits[key] = place_hits.get(key, 0.0) + weight
            anywhere[uid] = anywhere.get(uid, 0.0) + weight
            rank_moment[uid] = rank_moment.get(uid, 0.0) + weight * place

    presence = {uid: hits / total_weight for uid, hits in anywhere.items()}

    rows = []
    for place in range(1, PLACES + 1):
        shaped = {}
        for uid in sorted(presence):
            raw = (place_hits.get((place, uid), 0.0)
                   + SMOOTHING * presence[uid]) / (total_weight + SMOOTHING)
            if raw > 0.0:
                shaped[uid] = raw ** SHARPENING
        shaped_total = sum(shaped[uid] for uid in sorted(shaped))
        if shaped_total <= 0.0:
            continue
        scale = (1.0 - RESERVE) / shaped_total
        for uid in sorted(shaped):
            p = shaped[uid] * scale
            if p >= MIN_PROB:
                rows.append((uid, place, p))

    def top10_key(uid):
        mean_rank = rank_moment[uid] / anywhere[uid]
        return (-anywhere[uid], mean_rank, uid)

    top10 = sorted(presence, key=top10_key)[:PLACES]
    return rows, top10


def persistence_forecast(chart):
    """Fallback when no full top-10 read survives: trust the last chart.

    Half of each place goes to the game sitting there, 0.15 to each
    neighbouring game, and the rest stays unnamed, mirroring the idea
    that the chart moves slowly when there is nothing better to go on.
    """
    rows = []
    for place in range(1, PLACES + 1):
        centre = chart.get(place)
        if centre is None:
            continue
        rows.append((centre, place, 0.5))
        for neighbour in (place - 1, place + 1):
            other = chart.get(neighbour)
            if other is not None:
                rows.append((other, place, 0.15))
    top10 = [chart[place] for place in sorted(chart)][:PLACES]
    return rows, top10


def build_forecast(input_dir):
    """Return the forecast object with exactly the two required keys."""
    charts = read_charts(input_dir)
    full = [chart for chart in charts if len(chart) == PLACES]
    if full:
        rows, top10 = occupancy_forecast(full)
    elif charts:
        rows, top10 = persistence_forecast(charts[-1])
    else:
        rows, top10 = [], []
    probabilities = [
        {"universe_id": uid, "place": place, "p": round(p, 6)}
        for uid, place, p in rows
    ]
    probabilities.sort(key=lambda row: (row["place"], -row["p"],
                                        row["universe_id"]))
    return {"probabilities": probabilities, "top10": top10}


def check_forecast(forecast):
    """Refuse to write a forecast the scorer would treat as no forecast.

    A malformed file scores the floor at every place, so the shape the
    rules demand is verified here, before anything touches disk: the
    two keys and nothing else, at most 10 distinct top-10 ids, every
    probability inside (0, 1], at most 100 rows and a summed mass of at
    most 1 per place, and a payload inside the 1 MiB cap.
    """
    if set(forecast) != {"probabilities", "top10"}:
        raise ValueError("forecast must carry exactly probabilities, top10")
    top10 = forecast["top10"]
    if len(top10) > PLACES or len(set(top10)) != len(top10):
        raise ValueError("top10 must hold distinct ids, at most 10")
    if not all(isinstance(uid, int) for uid in top10):
        raise ValueError("top10 ids must be integers")
    mass = {}
    named = {}
    for row in forecast["probabilities"]:
        if not 0.0 < row["p"] <= 1.0:
            raise ValueError("probability outside (0, 1]")
        place = row["place"]
        mass[place] = mass.get(place, 0.0) + row["p"]
        named[place] = named.get(place, 0) + 1
        if mass[place] > 1.0 + 1e-9 or named[place] > 100:
            raise ValueError("place %d breaks a scoring cap" % place)
    payload = json.dumps(forecast, separators=(",", ":"))
    if len(payload.encode("utf-8")) > 1_048_576:
        raise ValueError("forecast exceeds the 1 MiB cap")


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    forecast = build_forecast(input_dir)
    check_forecast(forecast)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
