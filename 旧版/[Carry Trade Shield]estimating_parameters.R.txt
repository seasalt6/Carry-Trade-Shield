# Load the dplyr library for data manipulation
library(dplyr)

# Read CSV files containing bond yields and NASDAQ data, specifying not to convert strings to factors
us_bond <- read.csv("data/US 3-Month Bond Yield.csv", stringsAsFactors = FALSE)
jpn_bond <- read.csv("data/Japan 3-Month Bond Yield.csv", stringsAsFactors = FALSE)
nasdaq <- read.csv("data/NASDAQ.csv", stringsAsFactors = FALSE)

# Convert date columns to Date format with appropriate formats for each dataset
us_bond$Date <- as.Date(us_bond$DATE, format = "%Y/%m/%d")
jpn_bond$Date <- as.Date(jpn_bond$Date, format = "%m/%d/%Y")
nasdaq$Date <- as.Date(nasdaq$Date, format = "%m/%d/%Y")

# Sort the data by Date to ensure chronological order
us_bond <- us_bond[order(us_bond$Date), ]
jpn_bond <- jpn_bond[order(jpn_bond$Date), ]
nasdaq <- nasdaq[order(nasdaq$Date), ]

# Extract bond yield and compute log returns for Japan's bond yield and NASDAQ
us_bond_yield <- us_bond$DTB3  # Extract US 3-Month bond yield
jpn_bond_yield <- log(jpn_bond$Price / lag(jpn_bond$Price))  # Calculate log returns for Japan bond yield
jpn_bond_yield <- jpn_bond_yield[!is.na(jpn_bond_yield)]  # Remove NA values
jpn_bond_yield <- jpn_bond_yield[!is.infinite(jpn_bond_yield)]  # Remove infinite values
nasdaq_yield <- log(nasdaq$Close.Last / lag(nasdaq$Close.Last))  # Calculate NASDAQ log returns
nasdaq_yield <- nasdaq_yield[!is.na(nasdaq_yield)]  # Remove NA values

# Combine yields into a list for further analysis
yields <- list(us_bond_yield, jpn_bond_yield)

# Estimate parameters for each yield series
for (yield in yields) {
  theta_est <- mean(yield, na.rm = TRUE)  # Estimate long-term mean (theta)
  dY <- diff(yield, na.rm = TRUE)  # Compute changes in yield
  X_t_minus_theta <- yield[-length(yield)] - theta_est  # Deviation from mean
  kappa_est <- lm(dY ~ X_t_minus_theta)$coefficients[2]  # Estimate mean-reversion speed (kappa)
  sigma_est <- sd(residuals(lm(dY ~ X_t_minus_theta)))  # Estimate volatility (sigma)
  
  # Print the estimated parameters
  cat("For this series, estimated parameters are:\n")
  cat("κ (kappa):", kappa_est, "\n")
  cat("θ (theta):", theta_est, "\n")
  cat("σ (sigma):", sigma_est, "\n\n")
}

# Calculate daily volatility for NASDAQ using log returns
log_returns <- diff(log(nasdaq$Close.Last))  
sigma_S_daily <- sd(log_returns, na.rm = TRUE)  
cat("Estimated sigma_S:", sigma_S_daily, "\n")

# Read the exchange rate data and format the Date column
exchange_rate <- read.csv("data/USD-to-JPY Exchange Rate.csv", stringsAsFactors = FALSE)
exchange_rate$Date <- as.Date(exchange_rate$Date, format = "%m/%d/%Y")

# Align the lengths of Japan bond yield and US bond yield with exchange rate dates
dates <- exchange_rate$Date  

if (length(dates) > length(jpn_bond_yield)) {
  jpn_bond_yield <- c(jpn_bond_yield, rep(NA, length(dates) - length(jpn_bond_yield)))
} else if (length(dates) < length(jpn_bond_yield)) {
  jpn_bond_yield <- jpn_bond_yield[1:length(dates)]
}

jpn_bond_yield <- data.frame(Date = dates, Yield = jpn_bond_yield)
exchange_rate <- merge(exchange_rate, jpn_bond_yield, by = "Date", all.x = TRUE)

if (length(dates) > length(us_bond_yield)) {
  us_bond_yield <- c(us_bond_yield, rep(NA, length(dates) - length(us_bond_yield)))
} else if (length(dates) < length(us_bond_yield)) {
  us_bond_yield <- us_bond_yield[1:length(dates)]
}

us_bond_yield <- data.frame(Date = dates, Yield = us_bond_yield)
exchange_rate <- merge(exchange_rate, us_bond_yield, by = "Date", all.x = TRUE)

# Remove rows with NA or infinite values in exchange rate data
exchange_rate <- exchange_rate[!is.na(exchange_rate$Y) & !is.infinite(exchange_rate$Y), ]

# Calculate log of the exchange rate and its changes
exchange_rate$log_Y <- log(exchange_rate$Y)
exchange_rate$delta_log_Y <- c(NA, diff(exchange_rate$log_Y))

# Define the time step for daily data (1/252 trading days in a year)
dt <- 1 / 252

# Rename columns for clarity
colnames(exchange_rate)[colnames(exchange_rate) == "Yield.x"] <- "jpn_bond_yield"
colnames(exchange_rate)[colnames(exchange_rate) == "Yield.y"] <- "us_bond_yield"

# Calculate the drift term (difference in bond yields scaled by dt)
exchange_rate$drift <- (exchange_rate$jpn_bond_yield - exchange_rate$us_bond_yield) * dt

# Compute residuals between observed and drift-adjusted log exchange rate changes
exchange_rate$residuals <- exchange_rate$delta_log_Y - exchange_rate$drift

# Estimate volatility of the residuals (sigma_Y)
sigma_Y <- sd(exchange_rate$residuals, na.rm = TRUE) 
cat("Estimated sigma_Y:", sigma_Y, "\n")