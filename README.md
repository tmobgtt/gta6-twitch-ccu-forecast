# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tmobgtt/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tmobgtt/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A forecaster for Grand Theft Auto VI's launch-day Twitch audience. Given
everything recorded up to the end of the release day's first hour, it
writes, for each of hours 2 to 24 (UTC), five quantiles (5%, 25%, 50%,
75%, 95%) of that hour's mean concurrent viewers in the "Grand Theft
Auto VI" Twitch category.

The model is an empirical ratio model over the case's own comparables.
Each past big launch contributes its release day as log-ratios to its
own first hour. Launches are weighted by resemblance to the case at
hand: a Gaussian kernel on the log of the first hour's size (sigma
0.9), a Gaussian kernel on maturity, the log of hour 1 over the
previous day's peak (sigma 1.2), and a quarter weight when the unlock
shape differs (rolling midnight local versus one simultaneous global
unlock). Each hour's quantiles are the weighted quantiles of those
log-ratios, with the median curve smoothed over neighbouring hours
and the two tail quantiles blended halfway toward a fitted lognormal
tail; hours with too little comparable evidence fall back to a
persistence shape. The kernel widths were chosen by leave-one-out
backtests over the practice cases' comparables, where the model
scores a mean log pinball loss of about 0.25 on rolling-unlock
launches against the persistence baseline's 0.36. The program is a deterministic function
of its case file and uses only the Python 3.13 standard library.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
