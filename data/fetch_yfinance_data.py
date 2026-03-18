import pandas as pd
import yfinance as yf
from helpers import save_data, fill_nan

TICKER_TO_NAME = {
    "GC=F": "Gold Futures (COMEX)",
    "DX-Y.NYB": "U.S. Dollar Index",
    "EURUSD=X": "Euro / USD Exchange Rate",
    "JPY=X": "USD / JPY Exchange Rate",
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^VIX": "CBOE Volatility Index",
    "^MOVE": "ICE BofA MOVE Index",
    "CL=F": "WTI Crude Oil Futures",
    "BZ=F": "Brent Crude Oil Futures",
    "SI=F": "Silver Futures",
    "HG=F": "Copper Futures",
    "GLD": "SPDR Gold Shares ETF",
}


def download_yfinance_data(start="2006-01-01", end="2025-01-01", interval="1d", rename_columns=True):
    tickers = TICKER_TO_NAME.keys()
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,      # end is exclusive
        interval=interval,
        auto_adjust=True,
        group_by="column",
        threads=False,
        progress=False,
        keepna=False,
        repair=False,
        multi_level_index=True
    )
    if rename_columns and isinstance(df.columns, pd.MultiIndex):
        unique_tickers = df.columns.get_level_values(1).unique()
        new_names = [TICKER_TO_NAME.get(t, t) for t in unique_tickers]
        df.columns = df.columns.set_levels(new_names, level=1)
    return df

def print_price_types():
    """Print OHLCV price types with explanations."""
    price_types = [
        ("Open", "Price at the start of the trading day (market open)"),
        ("High", "Highest price reached during the day"),
        ("Low", "Lowest price reached during the day"),
        ("Close", "Price at the end of the trading day (market close)"),
        ("Volume", "Number of shares or contracts traded that day"),
    ]
    for name, desc in price_types:
        print(f"  {name:8} — {desc}")
    print()

def filter_by_price_types(df, price_types, flatten=False):
    """
    Return a 2D DataFrame with only columns for the given price types.

    Args:
        df: DataFrame with MultiIndex columns (Price, Ticker)
        price_types: List of price types to keep, e.g. ["Open"], ["Open", "Close"]
        flatten: If True, flatten MultiIndex to single-level columns. Default False.

    Returns:
        DataFrame with only columns matching the input price types
    """
    mask = df.columns.get_level_values(0).isin(price_types)
    result = df.loc[:, mask].copy()
    if flatten and len(price_types) == 1:
        result.columns = result.columns.get_level_values(1)
    return result

def save_data(df, filename):
    df.to_csv(filename, index=True)

def check_ticker_nan_consistency(df):
    """For each ticker, check if all price types (Open, High, Low, Close, Volume) have equal NaN counts."""
    nan_counts = df.isna().sum()
    for ticker in df.columns.get_level_values(1).unique():
        ticker_counts = nan_counts.xs(ticker, level=1)
        if ticker_counts.nunique() == 1:
            print(f"{ticker}: OK — all price types have {ticker_counts.iloc[0]} missing\n")
        else:
            print(f"{ticker}: MISMATCH — {ticker_counts.to_dict()}\n")

def print_statistics(df, price_types, columns):
    """Print range, mean, and other useful statistics for each column."""
    mask = (
        df.columns.get_level_values(0).isin(price_types)
        & df.columns.get_level_values(1).isin(columns)
    )
    filtered = df.loc[:, mask]
    print(filtered.describe().to_string())
    print()

def print_data_nan(df):
    print(f"Number of columns: {len(df.columns)}\n")
    print(f"Number of columns with missing values: {len(df.columns[df.isna().any()])}\n")
    print_price_types()
    print(f"Number of missing values:\n{df.isna().sum().to_string()}\n")
    print("Ticker NaN consistency (per price type):\n")
    check_ticker_nan_consistency(df)

def main():
    df = download_yfinance_data()
    save_data(df, "yfinance_data.csv")
    print_data_nan(df)
    df = fill_nan(df, method="ffill", second_method="bfill")
    print_data_nan(df)
    print_statistics(df, price_types=["Close"], columns=["Gold Futures (COMEX)"])
    save_data(df, "yfinance_data_filled.csv")
    df = filter_by_price_types(df, price_types=["Close"], flatten=True)
    save_data(df, "yfinance_data_filled_close.csv")

if __name__ == "__main__":
    main()