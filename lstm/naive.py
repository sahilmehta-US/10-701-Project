import pandas as pd
import numpy as np

df = pd.read_csv("../data/results/gold_base_stationary_dropna.csv", parse_dates=["Date"])
test = df[(df["Date"] >= "2022-01-01") & (df["Date"] <= "2024-12-31")]
y_true = test["Gold Futures (COMEX) | log_return"].values[1:]   # predict return of next day

# Zero forecast always pred 0
y_pred_zero = np.zeros_like(y_true)
dir_acc_zero = np.mean(np.sign(y_true) == np.sign(y_pred_zero))

# Persistence (predict same as previous day)
y_pred_pers = test["Gold Futures (COMEX) | log_return"].values[:-1]
dir_acc_pers = np.mean(np.sign(y_true) == np.sign(y_pred_pers))

# Positive bias (always predict +)
dir_acc_pos = np.mean(y_true > 0)

print(f"Zero forecast DirAcc: {dir_acc_zero:.3f}")
print(f"Persistence DirAcc:   {dir_acc_pers:.3f}")
print(f"Always-positive DirAcc: {dir_acc_pos:.3f}")