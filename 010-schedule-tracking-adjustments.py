# Explanation
# Sample Data Creation: The script creates a sample dataset containing project schedules and timeline data over a period of 100 days. This dataset is saved as schedule_tracking.csv.
# Data Loading and Plotting: The script reads the sample data from the CSV file and plots the project schedules and timeline data to visualize the time series.
# Train-Test Split: The data is split into training and testing sets for model evaluation.
# Model Definition and Training: An ARIMA model is defined and trained on the training set to forecast timeline data.
# Forecasting and Evaluation: The model forecasts timeline data for the test period. The forecast mean and confidence intervals are extracted. The mean squared error is calculated to evaluate the model's performance.
# Plotting the Forecast: The forecasted values are plotted alongside the actual values for the test period, with confidence intervals shown.
# Identifying Potential Delays: Residuals (differences between actual and forecasted values) are calculated, and positive residuals are identified as potential delays.
# Visualizing Potential Delays: A bar plot is created to visualize the identified potential delays.
# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn statsmodels


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Sample data creation
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
project_schedules = np.random.randint(5, 15, size=100)
timeline_data = project_schedules + np.random.randint(-3, 3, size=100)

# Create DataFrame
data = pd.DataFrame({'Date': dates, 'Project Schedules': project_schedules, 'Timeline Data': timeline_data})
data.set_index('Date', inplace=True)

# Save the sample data to a CSV file
data.to_csv('schedule_tracking.csv')

# Load the data
data = pd.read_csv('schedule_tracking.csv', index_col='Date', parse_dates=True)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Project Schedules'], label='Project Schedules')
plt.plot(data.index, data['Timeline Data'], label='Timeline Data')
plt.legend()
plt.title('Project Schedules and Timeline Data')
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define and train the ARIMA model
model = ARIMA(train['Timeline Data'], order=(5, 1, 0))
model_fit = model.fit()

# Forecasting
forecast = model_fit.get_forecast(steps=len(test))
forecast_index = test.index

# Get forecast mean, confidence intervals
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Evaluation
mse = mean_squared_error(test['Timeline Data'], forecast_mean)
print(f'Mean Squared Error: {mse}')

# Plotting the forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Timeline Data'], label='Train')
plt.plot(test.index, test['Timeline Data'], label='Test')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('Timeline Data Forecast')
plt.show()

# Identifying potential delays
residuals = test['Timeline Data'] - forecast_mean
potential_delays = residuals[residuals > 0]
print('Potential Delays:')
print(potential_delays)

# Visualizing potential delays
plt.figure(figsize=(12, 6))
sns.barplot(x=potential_delays.index, y=potential_delays.values)
plt.xticks(rotation=90)
plt.title('Potential Delays')
plt.xlabel('Date')
plt.ylabel('Delay Amount')
plt.show()
