# Load required libraries for data manipulation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Read CSV files containing bond yields and NASDAQ data
us_bond = pd.read_csv("data/US 3-Month Bond Yield.csv")
jpn_bond = pd.read_csv("data/Japan 3-Month Bond Yield.csv")
nasdaq = pd.read_csv("data/NASDAQ.csv")

# Convert date columns to Date format with appropriate formats for each dataset
us_bond["Date"] = pd.to_datetime(us_bond["DATE"], format="%Y/%m/%d")
jpn_bond["Date"] = pd.to_datetime(jpn_bond["Date"], format="%m/%d/%Y")
nasdaq["Date"] = pd.to_datetime(nasdaq["Date"], format="%m/%d/%Y")

# Sort the data by Date to ensure chronological order
us_bond = us_bond.sort_values("Date").reset_index(drop=True)
jpn_bond = jpn_bond.sort_values("Date").reset_index(drop=True)
nasdaq = nasdaq.sort_values("Date").reset_index(drop=True)

# Extract bond yield and compute log returns for Japan's bond yield and NASDAQ
us_bond_yield = us_bond["DTB3"]  # Extract US 3-Month bond yield
jpn_bond_yield = np.log(jpn_bond["Price"] / jpn_bond["Price"].shift(1))  # Calculate log returns for Japan bond yield
jpn_bond_yield = jpn_bond_yield.dropna()  # Remove NA values
jpn_bond_yield = jpn_bond_yield[~np.isinf(jpn_bond_yield)]  # Remove infinite values
nasdaq_yield = np.log(nasdaq["Close/Last"] / nasdaq["Close/Last"].shift(1))  # Calculate NASDAQ log returns
nasdaq_yield = nasdaq_yield.dropna()  # Remove NA values

# Combine yields into a list for further analysis
yields = [us_bond_yield, jpn_bond_yield]

# Estimate parameters for each yield series
for yield_series in yields:
    yield_series = yield_series.dropna().values
    theta_est = np.mean(yield_series)  # Estimate long-term mean (theta)
    dY = np.diff(yield_series)  # Compute changes in yield
    X_t_minus_theta = yield_series[:-1] - theta_est  # Deviation from mean
    # Estimate mean-reversion speed (kappa) via linear regression
    reg = LinearRegression().fit(X_t_minus_theta.reshape(-1, 1), dY)
    kappa_est = reg.coef_[0]
    sigma_est = np.std(dY - reg.predict(X_t_minus_theta.reshape(-1, 1)), ddof=1)  # Estimate volatility (sigma)

    # Print the estimated parameters
    print("For this series, estimated parameters are:")
    print("κ (kappa):", kappa_est)
    print("θ (theta):", theta_est)
    print("σ (sigma):", sigma_est)
    print()

# Calculate daily volatility for NASDAQ using log returns
log_returns = np.diff(np.log(nasdaq["Close/Last"]))
sigma_S_daily = np.std(log_returns[~np.isnan(log_returns)], ddof=1)
print("Estimated sigma_S:", sigma_S_daily)

# Read the exchange rate data and format the Date column
exchange_rate = pd.read_csv("data/USD-to-JPY Exchange Rate.csv")
exchange_rate["Date"] = pd.to_datetime(exchange_rate["Date"], format="%m/%d/%Y")

# Align the lengths of Japan bond yield and US bond yield with exchange rate dates
dates = exchange_rate["Date"]

jpn_bond_yield_values = jpn_bond_yield.values
if len(dates) > len(jpn_bond_yield_values):
    jpn_bond_yield_values = np.concatenate([jpn_bond_yield_values, np.full(len(dates) - len(jpn_bond_yield_values), np.nan)])
elif len(dates) < len(jpn_bond_yield_values):
    jpn_bond_yield_values = jpn_bond_yield_values[:len(dates)]

jpn_bond_yield_df = pd.DataFrame({"Date": dates, "Yield": jpn_bond_yield_values})
exchange_rate = exchange_rate.merge(jpn_bond_yield_df, on="Date", how="left")

us_bond_yield_values = us_bond_yield.values
if len(dates) > len(us_bond_yield_values):
    us_bond_yield_values = np.concatenate([us_bond_yield_values, np.full(len(dates) - len(us_bond_yield_values), np.nan)])
elif len(dates) < len(us_bond_yield_values):
    us_bond_yield_values = us_bond_yield_values[:len(dates)]

us_bond_yield_df = pd.DataFrame({"Date": dates, "Yield": us_bond_yield_values})
exchange_rate = exchange_rate.merge(us_bond_yield_df, on="Date", how="left", suffixes=("_x", "_y"))

# Remove rows with NA or infinite values in exchange rate data
exchange_rate = exchange_rate[exchange_rate["Y"].notna() & ~np.isinf(exchange_rate["Y"])].reset_index(drop=True)

# Calculate log of the exchange rate and its changes
exchange_rate["log_Y"] = np.log(exchange_rate["Y"])
exchange_rate["delta_log_Y"] = exchange_rate["log_Y"].diff()

# Define the time step for daily data (1/252 trading days in a year)
dt = 1 / 252

# Rename columns for clarity
exchange_rate = exchange_rate.rename(columns={"Yield_x": "jpn_bond_yield", "Yield_y": "us_bond_yield"})

# Calculate the drift term (difference in bond yields scaled by dt)
exchange_rate["drift"] = (exchange_rate["jpn_bond_yield"] - exchange_rate["us_bond_yield"]) * dt

# Compute residuals between observed and drift-adjusted log exchange rate changes
exchange_rate["residuals"] = exchange_rate["delta_log_Y"] - exchange_rate["drift"]

# Estimate volatility of the residuals (sigma_Y)
sigma_Y = exchange_rate["residuals"].std()
print("Estimated sigma_Y:", sigma_Y)
