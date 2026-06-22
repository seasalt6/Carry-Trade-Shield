import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_simulation(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0,
                           kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B):
    """Monte Carlo simulation for asset and exchange rate dynamics."""
    dt = T / M  # Time step
    Z = np.random.multivariate_normal(np.zeros(4), np.eye(4), size=N)  # Correlated random shocks

    def simulate_path():
        Y = Y0
        S = S0
        r_A_t = r_A
        r_B_t = r_B

        for _ in range(M):
            dW = np.sqrt(dt) * Z[np.random.choice(N, 4, replace=True)]  # Random shocks scaled by time step
            r_A_t = r_A_t + kappa_A * (theta_A - r_A_t) * dt + sigma_A * dW[0, 0]
            r_B_t = r_B_t + kappa_B * (theta_B - r_B_t) * dt + sigma_B * dW[1, 1]
            Y = Y + (r_B_t - r_A_t) * Y * dt + sigma_Y * Y * dW[2, 2]
            S = S + r_A_t * S * dt + sigma_S * S * dW[3, 3]

        return r_A_t, Y, S

    results = np.array([simulate_path() for _ in range(N)])  # Simulate N paths
    return results.T  # Shape: (3, N)


def calculate_payoff(dif, Strike_0, Strike_1, max_conp, principal):
    """Calculate payoff based on returns and strike levels."""
    payoffs = np.where(
        dif >= Strike_0,
        0,
        np.where(
            dif > Strike_1,
            -((Strike_0 - dif) / (Strike_0 - Strike_1)) * max_conp * dif * principal,
            -max_conp * dif * principal,
        ),
    )
    return payoffs


# Model parameters for interest rates, exchange rate, and stock price
r_A = 0.0441        # Initial interest rate for country A
kappa_A = 0.01299484
theta_A = 5.114246
sigma_A = 0.01674026

r_B = 0.0250         # Initial interest rate for country B
kappa_B = -1.235734
theta_B = -0.01198186
sigma_B = 0.3558819

sigma_Y = 0.00723001  # Volatility of exchange rate
sigma_S = 0.01098079  # Volatility of stock price

# Initial conditions and simulation parameters
Y0 = 150        # Initial exchange rate
S0 = 100        # Initial stock price
T = 0.25        # Time to maturity in years
N = 10_000_000  # Number of paths
M = 252         # Number of time steps

# Payoff structure parameters
Strike_0 = 0
Strike_1 = -0.05
max_conp = 0.5
principal = 1000

# Simulate paths and calculate returns
paths = monte_carlo_simulation(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0,
                               kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns = (np.log(paths[2]) + np.log(paths[1]) - np.log(Y0) - np.log(S0)) - r_B * T

plt.figure()
plt.hist(returns, bins=100, edgecolor='black')
plt.title("Distribution of Returns")
plt.show()

# Calculate payoffs
payoffs = calculate_payoff(returns, Strike_0, Strike_1, max_conp, principal)
plt.figure()
plt.hist(payoffs, bins=100, edgecolor='black')
plt.title("Distribution of Payoffs")
plt.show()

mean_payoffs = np.mean(payoffs)
print(f"Mean Payoffs: {mean_payoffs}")

# Bump stock price to calculate sensitivity (Delta for stock price)
h = 0.001

# Positive bump in stock price
paths_stock_pos = monte_carlo_simulation(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0 + 0.5 * h * S0,
                                         kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_stock_pos = (np.log(paths_stock_pos[2]) + np.log(paths_stock_pos[1]) - np.log(Y0) - np.log(S0)) - r_B * T
payoffs_stock_pos = calculate_payoff(returns_stock_pos, Strike_0, Strike_1, max_conp, principal)

# Negative bump in stock price
paths_stock_neg = monte_carlo_simulation(T, N, M, sigma_Y, Y0, r_A, r_B, sigma_S, S0 - 0.5 * h * S0,
                                         kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_stock_neg = (np.log(paths_stock_neg[2]) + np.log(paths_stock_neg[1]) - np.log(Y0) - np.log(S0)) - r_B * T
payoffs_stock_neg = calculate_payoff(returns_stock_neg, Strike_0, Strike_1, max_conp, principal)

delta_stock = (np.mean(payoffs_stock_pos) - np.mean(payoffs_stock_neg)) / (h * S0)
print(f"Delta Stock: {delta_stock}")

# Bump exchange rate to calculate sensitivity (Delta for exchange rate)
paths_fx_pos = monte_carlo_simulation(T, N, M, sigma_Y, Y0 + 0.5 * h * Y0, r_A, r_B, sigma_S, S0,
                                      kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_fx_pos = (np.log(paths_fx_pos[2]) + np.log(paths_fx_pos[1]) - np.log(Y0) - np.log(S0)) - r_B * T
payoffs_fx_pos = calculate_payoff(returns_fx_pos, Strike_0, Strike_1, max_conp, principal)

paths_fx_neg = monte_carlo_simulation(T, N, M, sigma_Y, Y0 - 0.5 * h * Y0, r_A, r_B, sigma_S, S0,
                                      kappa_A, theta_A, sigma_A, kappa_B, theta_B, sigma_B)
returns_fx_neg = (np.log(paths_fx_neg[2]) + np.log(paths_fx_neg[1]) - np.log(Y0) - np.log(S0)) - r_B * T
payoffs_fx_neg = calculate_payoff(returns_fx_neg, Strike_0, Strike_1, max_conp, principal)

delta_fx = (np.mean(payoffs_fx_pos) - np.mean(payoffs_fx_neg)) / (h * Y0)
print(f"Delta FX: {delta_fx}")
