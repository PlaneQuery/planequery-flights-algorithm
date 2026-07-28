import polars as pl
import pytest

from flights.flights_comparison import df_flights_comparison_stats


def test_flights_comparison_stats_include_absolute_airport_counts():
    comparison = pl.DataFrame(
        {
            "match_status": ["both", "both", "df0_only", "df1_only", "df1_only"],
            "same_takeoff_airport_ident": [True, False, False, False, False],
            "same_landing_airport_ident": [True, False, False, False, False],
            "same_airport_ident": [True, False, False, False, False],
        }
    )

    stats = df_flights_comparison_stats(comparison)

    assert stats["takeoff_airport_ident_total_count"] == 2
    assert stats["takeoff_airport_ident_match_count"] == 1
    assert stats["takeoff_airport_ident_incorrect_match_count"] == 1
    assert stats["landing_airport_ident_total_count"] == 2
    assert stats["landing_airport_ident_match_count"] == 1
    assert stats["landing_airport_ident_incorrect_match_count"] == 1
    assert stats["airport_ident_total_count"] == 2
    assert stats["airport_ident_match_count"] == 1
    assert stats["airport_ident_incorrect_match_count"] == 1
    assert stats["airport_ident_match_pct"] == pytest.approx(1 / 2)
