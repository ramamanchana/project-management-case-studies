import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sample Data
data = {
    'project_schedule': pd.date_range(start='1/1/2022', periods=5, freq='M'),
    'construction_activity_data': [100, 120, 130, 150, 170]
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index('project_schedule', inplace=True)

# Time-series Analysis for Schedule Management
model = ExponentialSmoothing(df['construction_activity_data'], trend='add', seasonal='add', seasonal_periods=12)
fit = model.fit()

# Prediction
predictions = fit.forecast(3)
print(f"Predicted Schedule Adherence: \n{predictions}")

# Plot
plt.plot(df.index, df['construction_activity_data'], label='Actual')
plt.plot(predictions.index, predictions, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Construction Activity')
plt.title('Construction Activity Prediction')
plt.legend()
plt.show()
