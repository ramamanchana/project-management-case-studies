# Explanation
# Sample Data Creation: The script creates a sample dataset containing business goals, design data, wellness standards, and strategic sustainability insights (misaligned or aligned). This dataset is saved as strategic_sustainability.csv.
# Data Loading and Preprocessing: The script reads the sample data from the CSV file and standardizes the features using StandardScaler.
# Feature Engineering and Preparation: The script prepares the input features (business goals, design data, and wellness standards) and the output target (strategic sustainability insights).
# Model Definition and Training: A Random Forest classifier is defined and trained to predict strategic sustainability insights.
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
business_goals = np.random.randint(1, 100, size=100)
design_data = np.random.randint(1, 100, size=100)
wellness_standards = np.random.randint(1, 100, size=100)
strategic_sustainability_insights = np.random.choice([0, 1], size=100)  # 0: Misaligned, 1: Aligned

# Create DataFrame
data = pd.DataFrame({
    'Business Goals': business_goals,
    'Design Data': design_data,
    'Wellness Standards': wellness_standards,
    'Strategic Sustainability Insights': strategic_sustainability_insights
})

# Save the sample data to a CSV file
data.to_csv('strategic_sustainability.csv', index=False)

# Load the data
data = pd.read_csv('strategic_sustainability.csv')

# Prepare input (X) and output (y)
X = data[['Business Goals', 'Design Data', 'Wellness Standards']]
y = data['Strategic Sustainability Insights']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting strategic sustainability insights
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
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Misaligned', 'Aligned'], yticklabels=['Misaligned', 'Aligned'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Get feature names
feature_names = ['Business Goals', 'Design Data', 'Wellness Standards']

# Determine the number of features to plot
num_features = len(feature_names)

plt.figure(figsize=(12, 6))
plt.title("Feature Importances for Strategic Sustainability Insights")
plt.bar(range(num_features), importances[indices], align="center")
plt.xticks(range(num_features), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, num_features])
plt.show()
