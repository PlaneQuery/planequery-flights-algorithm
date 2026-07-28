# planequery-flights-algorithm
**The world’s best ADS-B → flights algorithm.**

See [paper/Evaluating-and-Improving-Flight-Derivation-from-ADS-B-Data-Draft-Submission.pdf](paper/Evaluating-and-Improving-Flight-Derivation-from-ADS-B-Data-Draft-Submission-Draft-4.pdf)

Performance has been measured against [OpenSky’s](https://opensky-network.org/datasets/#trino-tables/flights/day=1772323200/) and [ADSBExchange’s](https://samples.adsbexchange.com/index.html#flights-ax-v2/2026/03/01) algorithms using [SWIM SFDPS](https://www.faa.gov/air_traffic/technology/swim/sfdps) and [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) flight data. It has also been tested against [EUROCONTROL](https://www.eurocontrol.int/dashboard/aviation-data-research) flight data. To assess the algorithm’s performance yourself, the repository includes flights generated from ADSB.lol ADS-B data from 2026-06-01 to 2026-07-01: [data/flights/algorithm/v1/adsblol/year=2026/month=06](data/flights/algorithm/v1/adsblol/year=2026/month=06).

The algorithm uses gradient-boosting ML for airport determination.

**Test date:** `2026-03-01 UTC`

| adsb_src | test_src  | gold_src | true_positive | false_positive | false_negative | precision | recall |     f1 | takeoff_airport_ident_match_pct | landing_airport_ident_match_pct | airport_ident_match_pct |
| -------- | --------- | -------- | ------------: | -------------: | -------------: | --------: | -----: | -----: | ------------------------------: | ------------------------------: | ----------------------: |
| adsb.lol | algorithm | sfdps    |         6,322 |             38 |             52 |    0.9940 | 0.9918 | 0.9929 |                          99.24% |                          98.54% |                  97.90% |
| adsb.lol | algorithm | bts      |         3,320 |              2 |             10 |    0.9994 | 0.9970 | 0.9982 |                          99.76% |                          99.73% |                  99.49% |
| adsbx    | algorithm | sfdps    |         6,357 |             61 |             17 |    0.9905 | 0.9973 | 0.9939 |                          99.91% |                          99.80% |                  99.72% |
| adsbx    | algorithm | bts      |         3,330 |              1 |              0 |    0.9997 | 1.0000 | 0.9998 |                         100.00% |                         100.00% |                 100.00% |
| adsbx    | adsbx     | sfdps    |         5,991 |             60 |            378 |    0.9901 | 0.9407 | 0.9647 |                          99.88% |                          99.92% |                  99.80% |
| adsbx    | adsbx     | bts      |         3,226 |              2 |            102 |    0.9994 | 0.9694 | 0.9841 |                         100.00% |                         100.00% |                 100.00% |
| opensky  | opensky   | sfdps    |         5,820 |            219 |            553 |    0.9637 | 0.9132 | 0.9378 |                          73.25% |                          67.47% |                  50.95% |
| opensky  | opensky   | bts      |         3,269 |             46 |             63 |    0.9861 | 0.9811 | 0.9836 |                          83.94% |                          75.41% |                  63.32% |

**Notes:**

* `precision`, `recall`, and `f1` are flight-segmentation metrics.
* Airport metrics are calculated only for flights that were successfully segmented.

## Eurocontrol data
| test_date  | adsb_src | test_src  | gold_src    | true_positive | false_positive | false_negative | precision | recall |     f1 | takeoff_airport_ident_match_pct | landing_airport_ident_match_pct | airport_ident_match_pct |
| ---------- | -------- | --------- | ----------- | ------------: | -------------: | -------------: | --------: | -----: | -----: | ------------------------------: | ------------------------------: | ----------------------: |
| 2024-06-01 | adsb.lol | algorithm | eurocontrol |         6,154 |             87 |            182 |    0.9861 | 0.9713 | 0.9786 |                          0.9594 |                          0.9366 |                  0.9007 |


# Setup
This repository uses Git LFS for model and data files.
```bash
# First install Git LFS using your platform's package manager or installer
git lfs install
git clone https://github.com/PlaneQuery/planequery-flights-algorithm.git
pip install .
```

# Evaluation

Downloads ~3GB from https://github.com/adsblol/globe_history_2026/releases and runs processing that takes ~5 minutes on M4 Macbook.

```bash
python src/flights/evaluation/evaluation.py \
  --adsb-src adsb.lol adsbx \
  --test-src algorithm adsbx opensky \
  --gold-src sfdps bts \
  --test-dates 2026-03-01
```

Note: ADSB.lol has reduced data coverage for 2026-06-11, which will result in worse performance for that day.
# Train Model

Downloads and processes 8 days of ADS-B data from https://github.com/adsblol/globe_history_2026/releases, ~24GB.

```bash
python src/flights_ai_model_airport/train.py
```

# Contributing:
All contributions welcome. Open a PR or issue with datasets or days you want to test algorithm on.

# AI Disclaimer

The airport model and testing apparatus used minimal AI assistance.  
The flight-segmentation algorithm used extensive AI assistance.
