import os
import requests
import pandas as pd
from dotenv import load_dotenv
from helpers import save_data, fill_nan

load_dotenv()

FRED_API_KEY = os.environ["FRED_API_KEY"]  # set this in .env file

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_ID_TO_NAME = {
    "DGS10": "U.S. 10-Year Treasury Yield",
    "DGS2": "U.S. 2-Year Treasury Yield",
    "T10Y2Y": "10Y Minus 2Y Treasury Spread",
    "FEDFUNDS": "Fed Funds Rate",
    "DFII10": "10Y Real Yield",
    "T10YIE": "10Y Breakeven Inflation",
    "M2SL": "M2 Money Supply",
}


def fetch_fred_series(
    series_id,
    start=None,
    end=None,
    units="lin",
    frequency=None,
    aggregation_method=None,
):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "units": units,
    }
    if start is not None:
        params["observation_start"] = start
    if end is not None:
        params["observation_end"] = end  # observation_end is inclusive
    if frequency is not None:
        params["frequency"] = frequency
    if aggregation_method is not None:
        params["aggregation_method"] = aggregation_method

    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["observations"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].rename(columns={"value": series_id})
    return df

def download_fred_data(start="2006-01-01", end="2024-12-31", rename_columns=True):
    """Download FRED series and return merged DataFrame with date index.
    Daily Treasury series define the calendar; monthly series are forward-filled onto it.
    """
    # Daily Treasury / market-macro series
    dgs10 = fetch_fred_series("DGS10", start=start, end=end)
    dgs2 = fetch_fred_series("DGS2", start=start, end=end)
    t10y2y = fetch_fred_series("T10Y2Y", start=start, end=end)
    dfii10 = fetch_fred_series("DFII10", start=start, end=end)
    t10yie = fetch_fred_series("T10YIE", start=start, end=end)

    # Merge daily series → defines the daily calendar
    fred_daily = (
        dgs10.merge(dgs2, on="date", how="outer")
        .merge(t10y2y, on="date", how="outer")
        .merge(dfii10, on="date", how="outer")
        .merge(t10yie, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_calendar = fred_daily[["date"]].copy()

    # Monthly series — forward-fill onto daily calendar
    fedfunds = fetch_fred_series("FEDFUNDS", start=start, end=end)
    m2sl = fetch_fred_series("M2SL", start=start, end=end)

    fedfunds_daily = daily_calendar.merge(fedfunds, on="date", how="left").sort_values("date")
    fedfunds_daily["FEDFUNDS"] = fedfunds_daily["FEDFUNDS"].ffill()

    m2sl_daily = daily_calendar.merge(m2sl, on="date", how="left").sort_values("date")
    m2sl_daily["M2SL"] = m2sl_daily["M2SL"].ffill()

    # Combine daily Treasury with daily-aligned monthly series
    df = (
        fred_daily
        .merge(fedfunds_daily[["date", "FEDFUNDS"]], on="date", how="left")
        .merge(m2sl_daily[["date", "M2SL"]], on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df.set_index("date")
    if rename_columns:
        df = df.rename(columns=FRED_ID_TO_NAME)
    return df

def print_data_nan(df):
    """Print NaN summary for DataFrame."""
    print(f"Number of columns: {len(df.columns)}\n")
    print(f"Number of columns with missing values: {len(df.columns[df.isna().any()])}\n")
    print(f"Number of missing values:\n{df.isna().sum().to_string()}\n")

def print_statistics(df, columns=None):
    """Print range, mean, and other useful statistics for each column."""
    if columns is not None:
        df = df[columns]
    print(df.describe().to_string())
    print()

def main():
    df = download_fred_data()
    save_data(df, "fred_data.csv")
    print_data_nan(df)
    df = fill_nan(df, method="ffill", second_method="bfill")
    print_data_nan(df)
    print_statistics(df, columns=["U.S. 10-Year Treasury Yield"])
    save_data(df, "fred_data_filled.csv")

if __name__ == "__main__":
    main()
