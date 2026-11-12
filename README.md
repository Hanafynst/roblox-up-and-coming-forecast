# Predict Roblox's Up-and-Coming top 10 — roblox-up-and-coming-forecast-01

Entry for the make.fun competition "Predict Roblox's Up-and-Coming top 10"
(roblox-up-and-coming-forecast-01). `main.py` reads the chart history and
forecasts each of the top 10 places with a recent-read empirical occupancy
method: the newest 14 reads are weighted on a 0.85 recency ramp, each place's
distribution is the weighted share of those reads a game actually spent at
that place — smoothed toward its overall top-10 presence, sharpened, with a
small reserve left unnamed and sub-1% rows dropped into that reserve.

Referral: https://make.fun/r/byvgg6sg
