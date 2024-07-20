import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample Data
data = {
    'change_management_plans': [1, 2, 3, 4, 5],
    'project_data': [10, 20, 30, 40, 50],
    'transition_smoothness': [1, 0, 1, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# AI-driven Change Management Model
X = df[['change_management_plans', 'project_data']]
y = df['transition_smoothness']

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'change_management_plans': [3],
    'project_data': [35]
})

# Prediction
predicted_transition = model.predict(new_data)
print(f"Predicted Transition Smoothness: {predicted_transition[0]}")
