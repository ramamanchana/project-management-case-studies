# Explanation
# Sample Data Creation: The script creates a sample dataset containing business objectives scores, market data scores, project requirements scores, and feasibility scores. This dataset is saved as needs_assessment.csv.
# Data Loading and Preprocessing: The script reads the sample data from the CSV file and standardizes the features using StandardScaler.
# Feature Engineering and Preparation: The script prepares the input features (business objectives score, market data score, and project requirements score) and the output target (feasibility score).
# Model Definition and Training: A Linear Regression model is defined and trained to predict feasibility scores.
# Model Evaluation: The model's performance is evaluated using mean squared error and R-squared metrics.
# Visualization: The results are visualized using a scatter plot of actual vs predicted values and a feature importance bar plot.

# Before running the script, ensure you have the necessary libraries installed:
# pip install pandas numpy matplotlib seaborn scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Sample data creation
data = {
    'Business Objectives Score': np.random.randint(1, 10, size=100),
    'Market Data Score': np.random.randint(1, 10, size=100),
    'Project Requirements Score': np.random.randint(1, 10, size=100),
    'Feasibility Score': np.random.randint(50, 100, size=100)  # Example feasibility scores
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('needs_assessment.csv', index=False)

# Load the data
data = pd.read_csv('needs_assessment.csv')

# Prepare input (X) and output (y)
X = data[['Business Objectives Score', 'Market Data Score', 'Project Requirements Score']]
y = data['Feasibility Score']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting feasibility scores
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizing the results
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Feasibility Score')
plt.ylabel('Predicted Feasibility Score')
plt.title('Actual vs Predicted Feasibility Scores')
plt.show()

# Feature Importance
coefficients = model.coef_
indices = np.argsort(np.abs(coefficients))[::-1]

# Get feature names
feature_names = ['Business Objectives Score', 'Market Data Score', 'Project Requirements Score']

# Determine the number of features to plot
num_features = len(feature_names)

plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Feasibility Scores")
plt.bar(range(num_features), coefficients[indices], align="center")
plt.xticks(range(num_features), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, num_features])
plt.show()
