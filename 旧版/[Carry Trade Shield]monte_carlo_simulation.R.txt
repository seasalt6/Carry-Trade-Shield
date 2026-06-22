# Load required libraries
library(ggplot2)  # For data visualization
library(MASS)  # For multivariate normal distribution

# Define a Monte Carlo simulation function for asset and exchange rate dynamics
monte_carlo_simulation <- function(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0, 
                                   kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B) {
  dt <- T / M  # Time step (length of each simulation step)
  Z <- mvrnorm(N, mu = rep(0, 4), Sigma = diag(4))  # Generate correlated random shocks
  
  # Function to simulate a single path
  simulate_path <- function() {
    Y <- Y0  # Initialize exchange rate
    S <- S0  # Initialize stock price
    r_A_t <- r_A  # Initialize interest rate for country A
    r_B_t <- r_B  # Initialize interest rate for country B
    
    # Simulate over M time steps
    for (t in seq_len(M)) {
      dW <- sqrt(dt) * Z[sample(N, 4, replace = TRUE), ]  # Random shocks scaled by time step
      r_A_t <- r_A_t + kappa_A * (theta_A - r_A_t) * dt + sigma_A * dW[1]  # Update r_A_t
      r_B_t <- r_B_t + kappa_B * (theta_B - r_B_t) * dt + sigma_B * dW[2]  # Update r_B_t
      Y <- Y + (r_B_t - r_A_t) * Y * dt + sigma_Y * Y * dW[3]  # Update exchange rate
      S <- S + r_A_t * S * dt + sigma_S * S * dW[4]  # Update stock price
    }
    
    c(r_A_t, Y, S)  # Return final values of interest rates, exchange rate, and stock price
  }
  replicate(N, simulate_path())  # Simulate N paths
}

# Define a function to calculate payoff based on returns and strike levels
calculate_payoff <- function(dif, Strike_0, Strike_1, max_conp, principal) {
  sapply(dif, function(x) {
    if (x >= Strike_0) {
      return(0)  # No loss if return is above Strike_0
    } else if (x > Strike_1) {
      return(-((Strike_0 - x) / (Strike_0 - Strike_1)) * max_conp * x * principal)  # Scaled loss
    } else {
      return(-max_conp * x * principal)  # Maximum loss
    }
  })
}

# Model parameters for interest rates, exchange rate, and stock price
r_A <- 0.0441  # Initial interest rate for country A
kappa_A <- 0.01299484  # Mean-reversion speed for r_A
theta_A <- 5.114246  # Long-term mean for r_A
sigma_A <- 0.01674026  # Volatility of r_A

r_B <- 0.0250  # Initial interest rate for country B
kappa_B <- -1.235734  # Mean-reversion speed for r_B
theta_B <- -0.01198186  # Long-term mean for r_B
sigma_B <- 0.3558819  # Volatility of r_B

sigma_Y <- 0.00723001  # Volatility of exchange rate
sigma_S <- 0.01098079  # Volatility of stock price

# Initial conditions and simulation parameters
Y0 <- 150  # Initial exchange rate
S0 <- 100  # Initial stock price
T <- 0.25  # Time to maturity in years
N <- 10000000  # Number of paths
M <- 252  # Number of time steps (daily steps for a year)

# Payoff structure parameters
Strike_0 <- 0  # Upper strike level
Strike_1 <- -0.05  # Lower strike level
max_conp <- 0.5  # Maximum payoff multiplier
principal <- 1000  # Principal amount

# Simulate paths and calculate returns
paths <- monte_carlo_simulation(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0, 
                                kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns <- (log(paths[3,]) + log(paths[2,]) - log(Y0) - log(S0)) - r_B * T  # Calculate returns
qplot(returns)  # Visualize the distribution of returns

# Calculate payoffs and visualize the payoff distribution
payoffs <- calculate_payoff(returns, Strike_0, Strike_1, max_conp, principal)
qplot(payoffs)
mean_payoffs <- mean(payoffs)  # Calculate mean payoff
cat("Mean Payoffs:", mean_payoffs, "\n")

# Bump stock price to calculate sensitivity (Delta for stock price)
h <- 0.001  # Small bump percentage

# Positive bump in stock price
paths_stock_pos <- monte_carlo_simulation(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0 + 0.5 * h * S0, 
                                          kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_stock_pos <- (log(paths_stock_pos[3,]) + log(paths_stock_pos[2,]) - log(Y0) - log(S0)) - r_B * T
payoffs_stock_pos <- calculate_payoff(returns_stock_pos, Strike_0, Strike_1, max_conp, principal)

# Negative bump in stock price
paths_stock_neg <- monte_carlo_simulation(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0 - 0.5 * h * S0, 
                                          kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_stock_neg <- (log(paths_stock_neg[3,]) + log(paths_stock_neg[2,]) - log(Y0) - log(S0)) - r_B * T
payoffs_stock_neg <- calculate_payoff(returns_stock_neg, Strike_0, Strike_1, max_conp, principal)

# Calculate Delta for stock price
delta_stock <- (mean(payoffs_stock_pos) - mean(payoffs_stock_neg)) / (h * S0)
cat("Delta Stock:", delta_stock, "\n")

# Bump exchange rate to calculate sensitivity (Delta for exchange rate)
paths_fx_pos <- monte_carlo_simulation(T, N, M, sigma_Y, Y0 + 0.5 * h * Y0, r_A, r_B, sigma_S, S0,
                                       kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_fx_pos <- (log(paths_fx_pos[3,]) + log(paths_fx_pos[2,]) - log(Y0) - log(S0)) - r_B * T
payoffs_fx_pos <- calculate_payoff(returns_fx_pos, Strike_0, Strike_1, max_conp, principal)

paths_fx_neg <- monte_carlo_simulation(T, N, M, sigma_Y, Y0 - 0.5 * h * Y0, r_A, r_B, sigma_S, S0, 
                                       kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_fx_neg <- (log(paths_fx_neg[3,]) + log(paths_fx_neg[2,]) - log(Y0) - log(S0)) - r_B * T
payoffs_fx_neg <- calculate_payoff(returns_fx_neg, Strike_0, Strike_1, max_conp, principal)

# Calculate Delta for exchange rate
delta_fx <- (mean(payoffs_fx_pos) - mean(payoffs_fx_neg)) / (h * Y0)
cat("Delta FX:", delta_fx, "\n")