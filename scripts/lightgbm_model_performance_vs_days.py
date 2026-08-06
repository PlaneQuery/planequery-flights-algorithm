from datetime import date, timedelta, datetime

import matplotlib.pyplot as plt
import polars as pl

from data_engineering.flights.flight_type import get_flights
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from data_engineering.utils import OUTPUT_DIR
from flights.evaluation.evaluation import read_adsb_for_matching
from flights.flights_comparison import df_flights_comparision, df_flights_comparison_stats
from flights.flights_match_adsb import get_matching_icaos_in_flights
from flights_ai_model_airport.features import TrainingDataCache
from flights_ai_model_airport.inference import run_inference
from flights_ai_model_airport.train import train

start_date = date(2026, 6, 1)
test_date = date(2026, 2, 1)
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = OUTPUT_DIR / "data/experiments/airport_model_performance_over_days" / f"{run_timestamp}_{test_date.isoformat()}"

rows: list[dict[str, int | float | str]] = []

max_training_days = 10

training_days_equivalents = [0.001, 0.05, 0.1, 0.25, 0.5, 0.75, *range(1, max_training_days + 1)]

training_dates_to_cache = [
    start_date + timedelta(days=i)
    for i in range(max_training_days)
]
training_data_cache = TrainingDataCache()
training_flight_counts_by_date = {}
for training_date in training_dates_to_cache:
    training_flight_counts_by_date[training_date] = len(
        training_data_cache.get_flights(training_date)
    )

same_day_expr = (
    (pl.col("takeoff_time").dt.date() == test_date)
    & (pl.col("landing_time").dt.date() == test_date)
)
df_sfdps_flights = get_sfdps_flights_day(test_date).filter(same_day_expr)
df_adsb = read_adsb_for_matching(
    df_sfdps_flights,
    test_date,
    adsb_src="adsbx",
    pia_or_american_ladd_only=False,
)
df_gold = get_matching_icaos_in_flights(
    df_sfdps_flights,
    df_adsb,
    datetime.combine(test_date, datetime.min.time()),
)
gold_icaos = df_gold.get_column("icao").drop_nulls().unique()
df_test_flights = get_flights(test_date).filter(
    same_day_expr & pl.col("icao").is_in(gold_icaos.implode())
)
print(
    f"Running inference on {len(df_test_flights):,} flights for "
    f"{len(gold_icaos):,} gold ICAOs"
)

for days_worth in training_days_equivalents:
    num_full_days = int(days_worth)
    num_dates = max(1, num_full_days)
    training_dates = [
        start_date + timedelta(days=i)
        for i in range(num_dates)
    ]
    if days_worth < 1:
        num_training_flights = max(
            1,
            round(training_flight_counts_by_date[start_date] * days_worth),
        )
    else:
        num_training_flights = sum(
            training_flight_counts_by_date[dt]
            for dt in training_dates
        )
    print(
        f"Training on {num_training_flights:,} flights "
        f"({days_worth:g} days worth): {training_dates}"
    )
    path = train(
        training_dates,
        max_training_flights=num_training_flights,
        training_data_cache=training_data_cache,
    )
    df_test = run_inference(
        test_date,
        model_path=path,
        df_flights=df_test_flights,
    )
    df = df_flights_comparision(df_test, df_gold, datetime(test_date.year, test_date.month, test_date.day), datetime(test_date.year, test_date.month, test_date.day) + timedelta(days=1), compare_airports=False)
    stats = df_flights_comparison_stats(df)
    rows.append(
        {
            "num_training_flights": num_training_flights,
            "training_days_equivalent": days_worth,
            "test_date": test_date.isoformat(),
            **stats,
        }
    )

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "lightgbm_model_performance_vs_days.csv"
results_df = pl.DataFrame(rows).sort("num_training_flights")
results_df.write_csv(output_path)
print(f"Saved performance stats to {output_path}")

num_training_flights = results_df.get_column("num_training_flights").to_list()
days_worth = results_df.get_column("training_days_equivalent").to_list()
x = list(range(len(results_df)))
fig, ax = plt.subplots(figsize=(14, 6))

metric_labels = {
    "takeoff_airport_ident_match_pct": "takeoff_airport_match_pct",
    "landing_airport_ident_match_pct": "landing_airport_match_pct",
    "airport_ident_match_pct": "airport_match_pct",
}
metric_values = {
    column: results_df.get_column(column).to_list()
    for column in metric_labels
}
for column, values in metric_values.items():
    ax.plot(x, values, marker="o", label=metric_labels[column])

ax.set_xlabel("Number of Training Flights (Days Worth)")
ax.set_ylabel("Airport Match Metrics")
observed_min = min(min(values) for values in metric_values.values())
ax.set_ylim(max(0, observed_min - 0.02), 1.0)
ax.set_xticks(x)
ax.set_xticklabels(
    [
        f"{num_flights:,}\n({num_days:g} {'day' if num_days == 1 else 'days'})"
        for num_flights, num_days in zip(num_training_flights, days_worth)
    ],
    rotation=45,
    ha="right",
)
ax.margins(x=0.03)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")

fig.suptitle(f"LightGBM Airport Model Performance vs Training Flights ({test_date.isoformat()})")
fig.tight_layout()
graph_output_path = output_dir / "lightgbm_model_performance_vs_days.png"
fig.savefig(graph_output_path, dpi=200)
plt.close(fig)
print(f"Saved performance graph to {graph_output_path}")
