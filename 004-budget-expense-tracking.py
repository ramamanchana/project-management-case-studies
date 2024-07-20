# Explanation
# Sample Data Creation: The script creates a sample dataset containing budget records, project expenses, and financial data over a period of 100 months. This dataset is saved as budget_forecasting.csv.
# Data Loading and Plotting: The script reads the sample data from the CSV file and plots the budget records, project expenses, and financial data to visualize the time series.
# Train-Test Split: The data is split into training and testing sets for model evaluation.
# Model Definition and Training: An ARIMA model is defined and trained on the training set to forecast project expenses.
# Forecasting and Evaluation: The model forecasts project expenses for the test period. The forecast mean and confidence intervals are extracted. The mean squared error is calculated to evaluate the model's performance.
# Plotting the Forecast: The forecasted values are plotted alongside the actual values for the test period, with confidence intervals shown.
# Identifying Cost-Saving Opportunities: Residuals (differences between actual and forecasted values) are calculated, and positive residuals are identified as cost-saving opportunities.
# Visualizing Cost-Saving Opportunities: A bar plot is created to visualize the identified cost-saving opportunities.

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
dates = pd.date_range(start='2022-01-01', periods=100, freq='M')
budget_records = np.random.randint(50000, 150000, size=100)
project_expenses = budget_records + np.random.randint(-10000, 10000, size=100)
financial_data = budget_records + np.random.randint(-5000, 5000, size=100)

# Create DataFrame
data = pd.DataFrame({'Date': dates, 'Budget Records': budget_records, 'Project Expenses': project_expenses, 'Financial Data': financial_data})
data.set_index('Date', inplace=True)

# Save the sample data to a CSV file
data.to_csv('budget_forecasting.csv')

# Load the data
data = pd.read_csv('budget_forecasting.csv', index_col='Date', parse_dates=True)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Budget Records'], label='Budget Records')
plt.plot(data.index, data['Project Expenses'], label='Project Expenses')
plt.plot(data.index, data['Financial Data'], label='Financial Data')
plt.legend()
plt.title('Budget Records, Project Expenses, and Financial Data')
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define and train the ARIMA model
model = ARIMA(train['Project Expenses'], order=(5, 1, 0))
model_fit = model.fit()

# Forecasting
forecast = model_fit.get_forecast(steps=len(test))
forecast_index = test.index

# Get forecast mean, confidence intervals
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Evaluation
mse = mean_squared_error(test['Project Expenses'], forecast_mean)
print(f'Mean Squared Error: {mse}')

# Plotting the forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Project Expenses'], label='Train')
plt.plot(test.index, test['Project Expenses'], label='Test')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('Project Expenses Forecast')
plt.show()

# Identifying cost-saving opportunities
residuals = test['Project Expenses'] - forecast_mean
cost_saving_opportunities = residuals[residuals > 0]
print('Cost-Saving Opportunities:')
print(cost_saving_opportunities)

# Visualizing cost-saving opportunities
plt.figure(figsize=(12, 6))
sns.barplot(x=cost_saving_opportunities.index, y=cost_saving_opportunities.values)
plt.xticks(rotation=90)
plt.title('Cost-Saving Opportunities')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()
