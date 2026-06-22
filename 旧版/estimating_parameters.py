import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Read CSV files containing bond yields and NASDAQ data
us_bond = pd.read_csv("data/US 3-Month Bond Yield.csv")
jpn_bond = pd.read_csv("data/Japan 3-Month Bond Yield.csv")
nasdaq = pd.read_csv("data/NASDAQ.csv")

# Convert date columns to datetime format
us_bond["Date"] = pd.to_datetime(us_bond["DATE"], format="%Y/%m/%d")
jpn_bond["Date"] = pd.to_datetime(jpn_bond["Date"], format="%m/%d/%Y")
nasdaq["Date"] = pd.to_datetime(nasdaq["Date"], format="%m/%d/%Y")

# Sort by Date
us_bond = us_bond.sort_values("Date").reset_index(drop=True)
jpn_bond = jpn_bond.sort_values("Date").reset_index(drop=True)
nasdaq = nasdaq.sort_values("Date").reset_index(drop=True)

# Extract bond yield and compute log returns
us_bond_yield = us_bond["DTB3"].values
jpn_bond_yield = np.log(jpn_bond["Price"] / jpn_bond["Price"].shift(1)).dropna()
jpn_bond_yield = jpn_bond_yield[np.isfinite(jpn_bond_yield)].values
nasdaq_yield = np.log(nasdaq["Close.Last"] / nasdaq["Close.Last"].shift(1)).dropna().values

# Combine yields into a list for further analysis
yields = [us_bond_yield, jpn_bond_yield]

# Estimate parameters for each yield series
for y in yields:
    # Remove NaN values for estimation
    y_clean = y[~np.isnan(y)].astype(float)
    theta_est = np.mean(y_clean)  # Estimate long-term mean (theta)
    dY = np.diff(y_clean)  # Compute changes in yield
    X_t_minus_theta = y_clean[:-1] - theta_est  # Deviation from mean

    # Linear regression: dY ~ X_t_minus_theta
    reg = LinearRegression().fit(X_t_minus_theta.reshape(-1, 1), dY)
    kappa_est = reg.coef_[0]  # Mean-reversion speed (kappa)
    residuals = dY - reg.predict(X_t_minus_theta.reshape(-1, 1))
    sigma_est = np.std(residuals, ddof=1)  # Volatility (sigma)

    print("For this series, estimated parameters are:")
    print(f"κ (kappa): {kappa_est}")
    print(f"θ (theta): {theta_est}")
    print(f"σ (sigma): {sigma_est}\n")

# Calculate daily volatility for NASDAQ using log returns
log_returns = np.diff(np.log(nasdaq["Close.Last"].dropna()))
sigma_S_daily = np.std(log_returns, ddof=1)
print(f"Estimated sigma_S: {sigma_S_daily}")

# Read the exchange rate data and format the Date column
exchange_rate = pd.read_csv("data/USD-to-JPY Exchange Rate.csv")
exchange_rate["Date"] = pd.to_datetime(exchange_rate["Date"], format="%m/%d/%Y")

# Align lengths of bond yields with exchange rate dates
dates = exchange_rate["Date"].values

# Align Japan bond yield
if len(dates) > len(jpn_bond_yield):
    jpn_bond_yield_aligned = np.concatenate([jpn_bond_yield, np.full(len(dates) - len(jpn_bond_yield), np.nan)])
else:
    jpn_bond_yield_aligned = jpn_bond_yield[:len(dates)]

jpn_yield_df = pd.DataFrame({"Date": dates, "Yield": jpn_bond_yield_aligned})
exchange_rate = exchange_rate.merge(jpn_yield_df, on="Date", how="left")

# Align US bond yield
if len(dates) > len(us_bond_yield):
    us_bond_yield_aligned = np.concatenate([us_bond_yield, np.full(len(dates) - len(us_bond_yield), np.nan)])
else:
    us_bond_yield_aligned = us_bond_yield[:len(dates)]

us_yield_df = pd.DataFrame({"Date": dates, "Yield": us_bond_yield_aligned})
exchange_rate = exchange_rate.merge(us_yield_df, on="Date", how="left", suffixes=("_jpn", "_us"))

# Remove rows with NA or infinite values in exchange rate column Y
exchange_rate = exchange_rate[exchange_rate["Y"].notna() & np.isfinite(exchange_rate["Y"])]

# Calculate log of the exchange rate and its changes
exchange_rate["log_Y"] = np.log(exchange_rate["Y"])
exchange_rate["delta_log_Y"] = exchange_rate["log_Y"].diff()

# Define the time step for daily data
dt = 1 / 252

# Rename columns for clarity
exchange_rate = exchange_rate.rename(columns={"Yield_jpn": "jpn_bond_yield", "Yield_us": "us_bond_yield"})

# Calculate the drift term
exchange_rate["drift"] = (exchange_rate["jpn_bond_yield"] - exchange_rate["us_bond_yield"]) * dt

# Compute residuals
exchange_rate["residuals"] = exchange_rate["delta_log_Y"] - exchange_rate["drift"]

# Estimate volatility of the residuals (sigma_Y)
sigma_Y = exchange_rate["residuals"].std()
print(f"Estimated sigma_Y: {sigma_Y}")
