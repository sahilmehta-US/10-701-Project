from fetch_fred_data import download_fred_data
from fetch_yfinance_data import download_yfinance_data, filter_by_price_types
from helpers import fill_nan, save_data


def main():
    yfinance_df = download_yfinance_data()
    yfinance_df = fill_nan(yfinance_df, method="ffill", second_method="bfill")
    yfinance_df = filter_by_price_types(yfinance_df, price_types=["Close"], flatten=True)
    print(len(yfinance_df))

    fred_df = download_fred_data()
    fred_df = fill_nan(fred_df, method="ffill", second_method="bfill")
    print(len(fred_df))

    merged_df = yfinance_df.join(fred_df, how="left")
    print(len(merged_df))
    save_data(merged_df, "merged_data.csv")


if __name__ == "__main__":
    main()