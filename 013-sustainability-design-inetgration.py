# Case Study: Sustainable Design Integration in Project Management

"""
Description:
Integrating sustainable design principles and wellness features into projects to achieve certifications such as LEED, BREEAM, or WELL. Predictive models evaluate design plans against certification criteria, ensuring that projects meet sustainability and wellness standards.

Model: Sustainability Assessment

Data Input: Design plans, sustainability standards

Prediction: Certification readiness, quality metrics

Recommended Model: Sustainability Assessment for quality assurance

Customer Value Benefits: Quality Assurance, Strategic Alignment

Use Case Implementation:
Ensuring that projects meet high sustainability and wellness standards not only promotes environmental responsibility but also enhances occupant well-being, contributing to the overall success and appeal of the projects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Sample data creation
np.random.seed(42)
design_plans = np.random.randint(1, 100, size=100)
sustainability_standards = np.random.randint(1, 100, size=100)
certification_readiness = np.random.choice([0, 1], size=100)  # 0: Not ready, 1: Ready

# Create DataFrame
data = pd.DataFrame({
    'Design Plans': design_plans,
    'Sustainability Standards': sustainability_standards,
    'Certification Readiness': certification_readiness
})

# Save the sample data to a CSV file
data.to_csv('sustainability_assessment.csv', index=False)

# Load the data
data = pd.read_csv('sustainability_assessment.csv')

# Prepare input (X) and output (y)
X = data[['Design Plans', 'Sustainability Standards']]
y = data['Certification Readiness']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC(),
    'KNN': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'MLP': MLPClassifier()
}

# Define hyperparameters for grid search
param_grid = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'DecisionTree': {'max_depth': [3, 5, 7, 10]},
    'RandomForest': {'n_estimators': [50, 100, 150]},
    'SVC': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]},
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
        grid_search = GridSearchCV(model, param_grid[name], cv=kf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    best_models[name] = best_model
    model_performance.append({'Model': name, 'Accuracy': accuracy})
    print(f'{name} - Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')

# Convert model performance to DataFrame
performance_df = pd.DataFrame(model_performance)

# Visualize model performance
plt.figure(figsize=(14, 7))
sns.barplot(x='Model', y='Accuracy', data=performance_df)
plt.title('Model Performance Comparison (Accuracy)')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# Visualizing the results of the best model
best_model_name = performance_df.loc[performance_df['Accuracy'].idxmax()]['Model']
best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Not ready', 'Ready'], yticklabels=['Not ready', 'Ready'])
plt.title(f'Confusion Matrix ({best_model_name})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance for the best model if applicable
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = ['Design Plans', 'Sustainability Standards']

    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Importances for {best_model_name}")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.show()
