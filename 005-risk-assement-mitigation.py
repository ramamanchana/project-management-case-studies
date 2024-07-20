import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# Sample Data
data = {
    'risk_data': [0.2, 0.5, 0.3, 0.7, 0.6],
    'regulatory_standards': [1, 0, 1, 0, 1],
    'project_plans': [5, 4, 3, 2, 1],
    'potential_risks': [1, 0, 1, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Risk Assessment Model
X = df[['risk_data', 'regulatory_standards', 'project_plans']]
y = df['potential_risks']

# Train Model
model = GradientBoostingClassifier()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'risk_data': [0.6],
    'regulatory_standards': [1],
    'project_plans': [2]
})

# Prediction
predicted_risks = model.predict(new_data)
print(f"Predicted Potential Risks: {predicted_risks[0]}")

# Plot
plt.bar(df.index, df['potential_risks'], color='blue', label='Actual')
plt.bar(new_data.index, predicted_risks, color='red', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Potential Risks')
plt.title('Risk Identification and Mitigation')
plt.legend()
plt.show()
