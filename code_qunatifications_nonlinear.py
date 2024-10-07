import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Data setup
data = {
    'Country': [
        'Iceland', 'Netherlands', 'Denmark', 'Germany', 'Malta',
        'Croatia', 'Finland', 'Estonia', 'Latvia', 'Lithuania',
        'Slovakia'
    ],
    'Rate': [
        0.91, 0.81, 0.88, 0.98, 0.74,
        0.89, 0.88, 0.873, 0.9, 0.89,
        0.92
    ],
    'Cost': [
        0.11, 0.1, 0.13, 0.15, 0.037,
        0.07, 0.1, 0.1, 0.1, 0.1,
        0.15
    ]
}

df = pd.DataFrame(data)

# Sigmoid function definition
def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))

# Fit the sigmoid function to data
popt, pcov = curve_fit(sigmoid, df['Cost'], df['Rate'], bounds=(0, [100, 1, 1]))

# Extract fitted parameters
a, b, c = popt
print(f"Fitted parameters: a={a}, b={b}, c={c}")

# Netherlands data
current_cost_nl = 0.10
current_rate_nl = 0.81

# Predicted return rate for the current deposit amount
predicted_rate_nl = sigmoid(current_cost_nl, a, b, c)
adjustment_factor = current_rate_nl / predicted_rate_nl
print(f"Adjustment factor for NL: {adjustment_factor:.4f}")

# Exponential decay function for the adjustment factor
def exponential_decay_adjustment(cost, base_factor, decay_rate):
    return 1 + (base_factor - 1) * np.exp(-decay_rate * cost)
base_factor = 1.02 
decay_rate = 010.0

# Adjusted prediction for NL
deposit_amounts = [0.11, 0.13, 0.15, 0.35]
predicted_rates_adjusted = [
    sigmoid(cost, a, b, c) * exponential_decay_adjustment(cost, adjustment_factor, decay_rate) 
    for cost in deposit_amounts
]

# Print adjusted results
for cost, rate in zip(deposit_amounts, predicted_rates_adjusted):
    print(f"Adjusted predicted return rate for a deposit of {cost:.2f}€: {rate:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Cost'], df['Rate'], color='blue', label='Data Points')
plt.plot(np.linspace(0, 0.16, 100), sigmoid(np.linspace(0, 0.16, 100), *popt), color='green', label='Sigmoid Regression Fit')
plt.scatter(current_cost_nl, current_rate_nl, color='red', label='Netherlands Actual')
plt.scatter(deposit_amounts, predicted_rates_adjusted, color='orange', label='Netherlands Predicted')
plt.title('Return Rate vs Deposit Amount for Glass Bottles')
plt.xlabel('Deposit Amount (in €)')
plt.ylabel('Return Rate (fraction of 1)')
plt.grid()
plt.legend()
plt.xlim(0, 0.16)
plt.ylim(0.6, 1.0)
plt.show()





