import pandas as pd
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
