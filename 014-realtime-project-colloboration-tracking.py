# Explanation
# Sample Data Creation: The script creates a sample dataset containing project data, collaboration tools, reporting standards, and operational efficiency (inefficient or efficient). This dataset is saved as real_time_project_tracking.csv.
# Data Loading and Preprocessing: The script reads the sample data from the CSV file and standardizes the features using StandardScaler.
# Feature Engineering and Preparation: The script prepares the input features (project data, collaboration tools, and reporting standards) and the output target (operational efficiency).
# Model Definition and Training: A Random Forest classifier is defined and trained to predict operational efficiency.
# Model Evaluation: The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.
# Visualization: The results are visualized using a confusion matrix heatmap and a feature importance bar plot.
# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib seaborn scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Sample data creation
np.random.seed(42)
project_data = np.random.randint(1, 100, size=100)
collaboration_tools = np.random.randint(1, 100, size=100)
reporting_standards = np.random.randint(1, 100, size=100)
operational_efficiency = np.random.choice([0, 1], size=100)  # 0: Inefficient, 1: Efficient

# Create DataFrame
data = pd.DataFrame({
    'Project Data': project_data,
    'Collaboration Tools': collaboration_tools,
    'Reporting Standards': reporting_standards,
    'Operational Efficiency': operational_efficiency
})

# Save the sample data to a CSV file
data.to_csv('real_time_project_tracking.csv', index=False)

# Load the data
data = pd.read_csv('real_time_project_tracking.csv')

# Prepare input (X) and output (y)
X = data[['Project Data', 'Collaboration Tools', 'Reporting Standards']]
y = data['Operational Efficiency']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting operational efficiency
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualizing the results
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Inefficient', 'Efficient'], yticklabels=['Inefficient', 'Efficient'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Get feature names
feature_names = ['Project Data', 'Collaboration Tools', 'Reporting Standards']

# Determine the number of features to plot
num_features = len(feature_names)

plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Operational Efficiency")
plt.bar(range(num_features), importances[indices], align="center")
plt.xticks(range(num_features), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, num_features])
plt.show()
