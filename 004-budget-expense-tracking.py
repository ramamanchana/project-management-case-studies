import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Sample Data
data = {
    'budget_records': [10000, 15000, 20000, 25000, 30000],
    'project_expenses': [9000, 14000, 19000, 23000, 29000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Time-series Analysis for Budget Control
model = ARIMA(df['project_expenses'], order=(5, 1, 0))
fit = model.fit(disp=0)

# Prediction
predictions = fit.forecast(steps=3)
print(f"Predicted Budget Forecasts: \n{predictions}")

# Plot
plt.plot(df.index, df['project_expenses'], label='Actual')
plt.plot(range(len(df), len(df) + len(predictions)), predictions, label='Predicted', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Project Expenses')
plt.title('Budget Forecasting')
plt.legend()
plt.show()
