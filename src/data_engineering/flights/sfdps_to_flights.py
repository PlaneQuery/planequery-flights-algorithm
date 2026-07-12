from pathlib import Path
import os
import polars as pl
from datetime import date
from data_engineering.utils import OUTPUT_DIR

def sfdps_flights_parquet_path(target_date: date) -> Path:
    return OUTPUT_DIR/ "data" / "flights" / "sfdps" / "v1" / f"year={target_date.year}" / f"month={target_date.month:02d}" / f"day={target_date.day:02d}" / "part-0.parquet"

def get_sfdps_flights_day(target_date: date) -> pl.DataFrame:
    parquet_path = sfdps_flights_parquet_path(target_date)
    return pl.read_parquet(parquet_path)


if __name__ == "__main__":
    current_day = date(2026, 3, 1)
    df_flights = get_sfdps_flights_day(current_day)
    print(df_flights)
    print(df_flights)
