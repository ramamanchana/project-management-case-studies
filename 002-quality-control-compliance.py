import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sample Data
data = {
    'construction_logs': [1, 2, 3, 4, 5],
    'quality_standards': [1, 0, 1, 0, 1],
    'project_plans': [5, 4, 3, 2, 1],
    'quality_compliance': [1, 0, 1, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# AI-based Model for Quality Assurance
X = df[['construction_logs', 'quality_standards', 'project_plans']]
y = df['quality_compliance']

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'construction_logs': [3],
    'quality_standards': [1],
    'project_plans': [3]
})

# Prediction
predicted_compliance = model.predict(new_data)
print(f"Predicted Quality Compliance: {predicted_compliance[0]}")
