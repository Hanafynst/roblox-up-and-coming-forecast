"""Predict Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson + /input/task.json and writes one static forecast
to /output/forecast.json: for each of the top-10 places, a probability for
each named game being at that place, plus our top 10 in order. The forecast is
scored against every hourly chart read of the 72 hours after entries close,
so it estimates each game's average occupancy of each place over the window.

Model, in one paragraph: every game carries a latent rank position, started
at its last observed rank. Its drift is its shrunken, recency-weighted rank
velocity. From the history's end we simulate the chart hour by hour for the
whole window: positions random-walk with rank-dependent diffusion (sigma
grows with sqrt(rank): the top of the chart is stickier than the middle),
drift decays geometrically, and a game whose 28-day eligibility ends inside
the window (details.created, when the history carries details) drops out at
that hour. The fraction of simulated
hours a game spends at each place is its probability there, scaled so each
place keeps a reserve unnamed: that reserve is what a game nobody named (a
new release, a long-shot climber) scores from when it lands, instead of the
floor. The top 10 is ordered by capped mean simulated rank. Everything is
seeded and sorted, so two runs on the same history are byte-identical.

Python 3.13, standard library only, no network.
"""

import json
import math
import os
import random
from datetime import datetime

PARAMS = dict(
    hl=20.0,          # EWMA half-life (hours) for a game's level
    hlv=30.0,         # recency half-life (hours) for velocity pairs
    shrink=6.0,       # velocity shrinkage: n / (n + shrink)
    vcap=0.30,        # cap on |drift|, ranks per hour
    tilt=0.0,         # player-growth tilt on drift (kept 0: no gain in tests)
    pool_rank=45,     # candidate pool: last rank or level within this
    rho=0.98,         # hourly decay of drift
    kappa=0.0,        # mean reversion to level (0: tests showed it hurts)
    sig_c=0.55,       # diffusion: sigma(rank) = sig_c * sqrt(rank) per sqrt-hour
    gamma=1.0,        # occupancy temperature (1: honest occupancy)
    reserve=0.15,     # unnamed share left at every place
    pmin=0.0015,      # rows below this join the unnamed share
    w_last=1.0,       # start position: the last observed rank itself
                      # (tests: blending in the level cost ~4 points)
    seed=20261113,
    paths=1500,
)


def _parse_ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def load_input(input_dir):
    with open(os.path.join(input_dir, 'task.json'), encoding='utf-8') as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, 'history.ndjson'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    for r in reads:
        r['_t'] = _parse_ts(r['at'])
        r['_rank'] = {row['universe_id']: row['rank'] for row in r['up_and_coming']}
    reads.sort(key=lambda r: r['_t'])
    return task, reads


def forecast(task, reads, P):
    places = int(task.get('places', 10))
    hours = int(task.get('window', {}).get('slots', 72))
    t_end = reads[-1]['_t'].timestamp()
    try:
        t0 = _parse_ts(task['window']['from']).timestamp()
    except Exception:
        t0 = t_end

    # ---- per-game levels and drifts -------------------------------------
    games = {}
    for r in reads:
        t = (r['_t'].timestamp() - t_end) / 3600.0
        for row in r['up_and_coming']:
            g = games.setdefault(row['universe_id'], {'obs': [], 'play': []})
            g['obs'].append((t, row['rank']))
            if row.get('playing'):
                g['play'].append((t, math.log(row['playing'])))
    last = reads[-1]
    last_rank = dict(last['_rank'])
    chart_n = len(last['up_and_coming'])
    off = chart_n + 12.0
    all_t = [(r['_t'].timestamp() - t_end) / 3600.0 for r in reads]

    info = {}
    for g, d in games.items():
        obsmap = dict(d['obs'])
        lvl = None
        prev_t = None
        for t in all_t:
            rank = obsmap.get(t, off)
            if lvl is None:
                lvl = float(rank)
            else:
                a = 1 - math.exp(-max(t - prev_t, 0.01) / P['hl'])
                lvl = lvl + a * (rank - lvl)
            prev_t = t
        num = den = 0.0
        obs = d['obs']
        for (t1, r1), (t2, r2) in zip(obs, obs[1:]):
            dt = max(t2 - t1, 0.25)
            w = math.exp(t2 / P['hlv'])
            num += w * (r2 - r1) / dt
            den += w
        vel = (num / den) if den else 0.0
        vel *= len(obs) / (len(obs) + P['shrink'])
        vel = max(-P['vcap'], min(P['vcap'], vel))
        info[g] = {'lvl': lvl, 'drift': vel, 'last': last_rank.get(g)}

    pool = []
    for g, inf in info.items():
        recent_best = min(r for _, r in games[g]['obs'][-3:])
        if (inf['last'] is not None and inf['last'] <= P['pool_rank']) \
                or recent_best <= 30 or inf['lvl'] <= P['pool_rank']:
            pool.append(g)
    pool.sort()
    if len(pool) < places:
        pool = sorted(last_rank, key=lambda g: last_rank[g])[:40]

    # ---- 28-day eligibility expiries, when details are present ----------
    expiry = {}
    det = None
    for r in reversed(reads):
        if r.get('details'):
            det = r['details']
            break
    if det:
        for g in pool:
            d = det.get(str(g))
            if d and d.get('created'):
                try:
                    cr = _parse_ts(d['created'])
                    exp_h = (cr.timestamp() + 28 * 86400 - t0) / 3600.0
                    if 0 < exp_h < hours:
                        expiry[g] = exp_h
                except Exception:
                    pass

    # ---- simulate the window --------------------------------------------
    rng = random.Random(P['seed'])
    occ = {g: [0.0] * places for g in pool}
    cap_sum = {g: 0.0 for g in pool}
    n_top = {g: 0 for g in pool}
    x0 = []
    for g in pool:
        inf = info[g]
        x0.append(inf['lvl'] if inf['last'] is None
                  else P['w_last'] * inf['last'] + (1 - P['w_last']) * inf['lvl'])
    paths = P['paths']
    for _ in range(paths):
        x = list(x0)
        alive = [True] * len(pool)
        for h in range(hours):
            order = sorted(range(len(pool)), key=lambda i: (x[i], pool[i]))
            seen = 0
            for i in order:
                if not alive[i]:
                    continue
                seen += 1
                if seen <= places:
                    occ[pool[i]][seen - 1] += 1.0
                    n_top[pool[i]] += 1
                cap_sum[pool[i]] += min(seen, 15)
            dh = P['rho'] ** h
            for i, g in enumerate(pool):
                if not alive[i]:
                    continue
                if g in expiry and h + 1 >= expiry[g]:
                    alive[i] = False
                    x[i] = 1e9
                    continue
                inf = info[g]
                sig = P['sig_c'] * math.sqrt(max(x[i], 1.0))
                x[i] += inf['drift'] * dh + rng.gauss(0.0, sig)
                if x[i] < 0.2:
                    x[i] = 0.2
    tot = float(paths * hours)

    rows = []
    for k in range(1, places + 1):
        named = []
        for g in pool:
            p = occ[g][k - 1] / tot
            if p > 0:
                named.append([g, p])
        s = sum(p for _, p in named)
        if s > 0:
            scale = (1.0 - P['reserve']) / s
            named = [[g, p * scale] for g, p in named]
        for g, p in named:
            if p >= P['pmin']:
                rows.append({'universe_id': g, 'place': k, 'p': round(p, 6)})
    rows.sort(key=lambda r: (r['place'], -r['p'], r['universe_id']))

    top10 = sorted(pool, key=lambda g: (cap_sum[g] / tot, -n_top[g], g))[:places]
    return {'probabilities': rows, 'top10': top10}


def fallback_forecast(task, reads):
    """Persistence smear, if anything above fails: always write a valid file."""
    places = int(task.get('places', 10))
    chart = [row['universe_id'] for row in
             sorted(reads[-1]['up_and_coming'], key=lambda x: x['rank'])]
    probs = []
    for place in range(1, places + 1):
        if place > len(chart):
            break
        probs.append({'universe_id': chart[place - 1], 'place': place, 'p': 0.5})
        if place == 1:
            if len(chart) > 1:
                probs.append({'universe_id': chart[1], 'place': 1, 'p': 0.3})
            continue
        probs.append({'universe_id': chart[place - 2], 'place': place, 'p': 0.15})
        if place < len(chart):
            probs.append({'universe_id': chart[place], 'place': place, 'p': 0.15})
    return {'probabilities': probs, 'top10': chart[:places]}


def main():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    task, reads = load_input(input_dir)
    try:
        fc = forecast(task, reads, PARAMS)
        # self-check the shape the scorer demands
        sums = {}
        counts = {}
        for r in fc['probabilities']:
            sums[r['place']] = sums.get(r['place'], 0.0) + r['p']
            counts[r['place']] = counts.get(r['place'], 0) + 1
        assert all(v <= 1.0 + 1e-9 for v in sums.values())
        assert all(v <= 100 for v in counts.values())
        assert len(fc['top10']) == int(task.get('places', 10))
        assert len(set(fc['top10'])) == len(fc['top10'])
    except Exception:
        fc = fallback_forecast(task, reads)
    os.makedirs(output_dir, exist_ok=True)
    out_name = task.get('output', 'forecast.json')
    with open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as f:
        f.write(json.dumps(fc, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
