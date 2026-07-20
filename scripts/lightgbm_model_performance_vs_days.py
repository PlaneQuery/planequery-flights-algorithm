from datetime import date, timedelta, datetime

import matplotlib.pyplot as plt
import polars as pl

from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from data_engineering.utils import OUTPUT_DIR
from flights.flights_comparison import df_flights_comparision, df_flights_comparison_stats
from flights.flights_match_adsb import ADSB_MATCHING_COLUMNS, get_matching_icaos_in_flights
from flights_ai_model_airport.inference import run_inference
from flights_ai_model_airport.train import train

start_date = date(2026, 6, 1)
test_date = date(2026, 2, 1)
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = OUTPUT_DIR / "data/experiments/airport_model_performance_over_days" / f"{run_timestamp}_{test_date.isoformat()}"

rows: list[dict[str, int | float | str]] = []

for num_days in range(1, 16):
    training_dates = [
        start_date + timedelta(days=i)
        for i in range(num_days)
    ]
    print(f"Training for {num_days} days: {training_dates}")
    path = train(training_dates)
    df_test = run_inference(test_date, model_path=path)
    df_sfdps_flights = get_sfdps_flights_day(test_date)
    df_adsb = read_adsb(
        test_date,
        icaos=df_sfdps_flights.get_column("icao").drop_nulls().unique().to_list(),
        columns=ADSB_MATCHING_COLUMNS,
    )
    df_gold = get_matching_icaos_in_flights(df_sfdps_flights, df_adsb)
    df = df_flights_comparision(df_test, df_gold, datetime(test_date.year, test_date.month, test_date.day), datetime(test_date.year, test_date.month, test_date.day) + timedelta(days=1), compare_airports=False)
    stats = df_flights_comparison_stats(df)
    rows.append({"num_training_days": num_days, "test_date": test_date.isoformat(), **stats})

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "lightgbm_model_performance_vs_days.csv"
results_df = pl.DataFrame(rows).sort("num_training_days")
results_df.write_csv(output_path)
print(f"Saved performance stats to {output_path}")

x = results_df.get_column("num_training_days").to_list()
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, results_df.get_column("takeoff_airport_ident_match_pct").to_list(), marker="o", label="takeoff_airport_ident_match_pct")
ax.plot(x, results_df.get_column("landing_airport_ident_match_pct").to_list(), marker="o", label="landing_airport_ident_match_pct")
ax.plot(x, results_df.get_column("airport_ident_match_pct").to_list(), marker="o", label="airport_ident_match_pct")
ax.set_xlabel("Number of Training Days")
ax.set_ylabel("Airport Match Metrics")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend()

fig.suptitle(f"LightGBM Airport Model Performance vs Training Days ({test_date.isoformat()})")
fig.tight_layout()
graph_output_path = output_dir / "lightgbm_model_performance_vs_days.png"
fig.savefig(graph_output_path, dpi=200)
plt.close(fig)
print(f"Saved performance graph to {graph_output_path}")
