from datetime import date, datetime

import polars as pl

from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.utils import OUTPUT_DIR
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights.flights_match_adsb import ADSB_MATCHING_COLUMNS, get_matching_icaos_in_flights

test_date = date(2026, 3, 1)
output_dir = OUTPUT_DIR / "data/experiments/testset_to_gold"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_path = output_dir / f"{timestamp}_testset_to_gold_kept.csv"

ADSB_SRCS = ["adsblol", "opensky", "adsbx"]


def get_kept_stats(source_flight, adsb_src: str):
    df_source_flights = source_flight(test_date)
    total_flights = df_source_flights.height
    total_icaos = df_source_flights.get_column("icao").n_unique()
    df_adsb = read_adsb(
        test_date,
        icaos=df_source_flights.get_column("icao").drop_nulls().unique().to_list(),
        columns=ADSB_MATCHING_COLUMNS,
        source=adsb_src,
    )
    df_gold = get_matching_icaos_in_flights(
        df_source_flights,
        df_adsb,
        datetime.combine(test_date, datetime.min.time()),
    )
    gold_icaos = df_gold.get_column("icao").n_unique()
    kept_flights_percent = df_gold.height / total_flights * 100
    kept_icaos_percent = gold_icaos / total_icaos * 100
    return (
        total_flights,
        df_gold.height,
        kept_flights_percent,
        total_icaos,
        gold_icaos,
        kept_icaos_percent,
    )

source_flights = {
    "sfdps": get_sfdps_flights_day,
    "bts": get_bts_flights_for_day,
}

rows = []
for source_name, source_flight in source_flights.items():
    for adsb_src in ADSB_SRCS:
        (
            total_flights,
            gold_flights,
            kept_flights_percent,
            total_icaos,
            gold_icaos,
            kept_icaos_percent,
        ) = get_kept_stats(source_flight, adsb_src)
        rows.append(
            {
                "source": source_name,
                "adsb_src": adsb_src,
                "test_date": test_date.isoformat(),
                "total_flights": total_flights,
                "gold_flights": gold_flights,
                "kept_flights_percent": kept_flights_percent,
                "total_icaos": total_icaos,
                "gold_icaos": gold_icaos,
                "kept_icaos_percent": kept_icaos_percent,
            }
        )
        print(
            f"Kept percent for {source_name} using {adsb_src}: "
            f"{kept_flights_percent:.2f}% | "
            f"Kept ICAOs: {gold_icaos}/{total_icaos} ({kept_icaos_percent:.2f}%)"
        )

output_dir.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows).write_csv(output_path)
print(f"Saved kept results to {output_path}")
