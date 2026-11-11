# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/Hanafynst/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/Hanafynst/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A probabilistic forecaster for Roblox's Up-and-Coming chart. It reads the
chart's history (`history.ndjson` + `task.json`) and writes one static
forecast of the top 10 at every hour of the 72 hours after entries close:
for each place, a probability for each named game holding it, plus a top 10
in order.

How it works: every game carries a latent rank position, started at its last
observed rank, with a drift equal to its shrunken, recency-weighted rank
velocity. The chart is then simulated hour by hour over the window:
positions random-walk with rank-dependent diffusion (sigma grows with the
square root of rank — the top of the chart is stickier than the middle),
drift decays geometrically, and a game whose 28-day eligibility ends inside
the window drops out at that hour when the history carries `details`. The
share of simulated hours a game spends at each place is its probability
there, scaled so every place keeps a 15% unnamed reserve — that reserve is
what an unnamed game (a new release, a long-shot climber) scores from when
it lands. Parameters were chosen by backtesting against the house
persistence baseline on rolling 72-hour folds of the practice history
(mean log score −37.3 vs the baseline's −49.5 on the same folds, and ahead
on top-10 hits and agreement too). Pure Python 3.13 standard library, no
network, fully seeded: two runs on the same history are byte-identical.

## Run it

```sh
mkdir -p in out
# put the competition's task.json and history.ndjson in ./in
MAKEFUN_INPUT_DIR=in MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=byvgg6sg&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
