import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# 1. Load Data and Calculate Returns
df = pd.read_csv('btc_history.csv')
# GARCH models work on percentage returns, usually scaled by 100
df['returns'] = 100 * df['close'].pct_change().dropna()
returns = df['returns'].dropna()

# 2. Define the GARCH(1,1) Model
# p=1: use 1 lag of variance, q=1: use 1 lag of squared returns
model = arch_model(returns, vol='Garch', p=1, q=1)

# 3. Fit the Model
res = model.fit(disp='off')

# 4. Forecast Tomorrow's Volatility
forecasts = res.forecast(horizon=1)
# Square root of variance gives us the standard deviation (Volatility)
forecasted_vol = np.sqrt(forecasts.variance.values[-1, :][0])

print("--- GARCH Volatility Report ---")
print(res.summary().tables[1]) # Show the model coefficients
print(f"\nForecasted Volatility for next period: {forecasted_vol:.4f}%")

# 5. Logic: Risk-Adjusted Decision
if forecasted_vol > returns.std():
    print("⚠️ HIGH VOLATILITY REGIME: Reduce position sizes.")
else:
    print("✅ LOW VOLATILITY REGIME: Normal trading conditions.")

# 6. Optional Visualization
res.conditional_volatility.plot(title="Conditional Volatility (GARCH)")
plt.show()