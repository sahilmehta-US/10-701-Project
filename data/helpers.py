import pandas as pd

def save_data(df, filename):
    df.to_csv(filename, index=True)

def fill_nan(df, method="ffill", second_method="bfill", columns=None):
    """
    Fill NaN using forward fill, then backward fill for any remaining (e.g. leading) NaNs.

    Args:
        df: DataFrame to fill
        method: First fill method ("ffill" or "bfill")
        second_method: Second fill method for remaining NaNs, or None to skip
        columns: Columns to fill. If None, fill all columns.
    """
    cols_to_fill = columns if columns is not None else df.columns
    subset = df[cols_to_fill]
    filled = getattr(subset, method)()
    if second_method is not None:
        filled = getattr(filled, second_method)()
    new_df = df.copy()
    new_df[cols_to_fill] = filled
    return new_df