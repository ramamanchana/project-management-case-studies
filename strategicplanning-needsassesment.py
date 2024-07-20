import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample Data
data = {
    'business_objectives': [1, 2, 3, 4, 5],
    'market_data': [10, 20, 30, 40, 50],
    'project_requirements': [100, 200, 300, 400, 500],
    'project_feasibility': [50, 100, 150, 200, 250]
}

# Create DataFrame
df = pd.DataFrame(data)

# Predictive Analytics Model
X = df[['business_objectives', 'market_data', 'project_requirements']]
y = df['project_feasibility']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'business_objectives': [3],
    'market_data': [35],
    'project_requirements': [250]
})

# Prediction
predicted_feasibility = model.predict(new_data)
print(f"Predicted Project Feasibility: {predicted_feasibility[0]}")
