# Case Study: Needs Assessment and Feasibility Studies in Real Estate Project Management

"""
Description:
Using predictive analytics models to conduct needs assessments and feasibility studies, aligning real estate projects with broader business objectives.
This ensures that all projects are initiated based on solid data and strategic alignment, reducing the risk of misaligned projects and improving the chances of successful outcomes.

Model: Predictive Analytics

Data Input: Business objectives, market data, project requirements

Prediction: Project feasibility, strategic alignment

Recommended Model: Predictive Analytics for strategic planning

Customer Value Benefits: Cost Savings, Strategic Alignment

Use Case Implementation:
By applying predictive analytics, project managers can identify potential issues early, evaluate project feasibility more accurately,
and ensure that projects align with long-term business strategies, ultimately leading to more successful and cost-effective projects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns

# Sample data creation
np.random.seed(42)
business_objectives = np.random.randint(1, 10, size=100)
market_data = np.random.randint(1, 10, size=100)
project_requirements = np.random.randint(1, 10, size=100)
project_feasibility = np.random.randint(50, 100, size=100)

# Create DataFrame
data = pd.DataFrame({
    'Business Objectives': business_objectives,
    'Market Data': market_data,
    'Project Requirements': project_requirements,
    'Project Feasibility': project_feasibility
})

# Save the sample data to a CSV file
data.to_csv('needs_assessment.csv', index=False)

# Load the data
data = pd.read_csv('needs_assessment.csv')

# Prepare input (X) and output (y)
X = data[['Business Objectives', 'Market Data', 'Project Requirements']]
y = data['Project Feasibility']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'XGBoost': XGBRegressor(),
    'MLP': MLPRegressor()
}

# Define hyperparameters for grid search
param_grid = {
    'DecisionTree': {'max_depth': [3, 5, 7, 10]},
    'RandomForest': {'n_estimators': [50, 100, 150]},
    'SVR': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]},
    'KNN': {'n_neighbors': [3, 5, 7]},
    'GradientBoosting': {'n_estimators': [50, 100, 150]},
    'AdaBoost': {'n_estimators': [50, 100, 150]},
    'XGBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
}

# Evaluate models with k-fold cross-validation and grid search
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
model_performance = []

for name, model in models.items():
    print(f'Evaluating {name}...')
    if name in param_grid:
        grid_search = GridSearchCV(model, param_grid[name], cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    best_models[name] = best_model
    model_performance.append({'Model': name, 'MSE': mse, 'R2': r2})
    print(f'{name} - MSE: {mse}, R2: {r2}')

# Convert model performance to DataFrame
performance_df = pd.DataFrame(model_performance)

# Visualize model performance
plt.figure(figsize=(14, 7))
sns.barplot(x='Model', y='R2', data=performance_df)
plt.title('Model Performance Comparison (R2 Score)')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.show()

# Visualizing the results of the best model
best_model_name = performance_df.loc[performance_df['R2'].idxmax()]['Model']
best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Feasibility Score')
plt.ylabel('Predicted Feasibility Score')
plt.title(f'Actual vs Predicted Feasibility Scores ({best_model_name})')
plt.show()

# Feature Importance for the best model if applicable
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = ['Business Objectives', 'Market Data', 'Project Requirements']

    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Importances for {best_model_name}")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.show()
