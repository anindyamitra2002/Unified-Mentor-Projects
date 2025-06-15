# heart_disease_prediction.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
# Assumes 'dataset.csv' has columns including 'age', 'sex', 'chest pain type', ..., and 'target' for heart disease label
df = pd.read_csv('dataset.csv')

# Rename columns if necessary (for columns with spaces)
# Example: df.rename(columns={'chest pain type': 'chest_pain_type'}, inplace=True)

drop_cols = []  # any columns to drop (e.g., ID)

# Features and target
y = df['target']
X = df.drop(drop_cols + ['target'], axis=1)

# One-hot encode categorical features
categorical_cols = [c for c in X.columns if X[c].dtype == 'object' or df[c].nunique() < 10]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define model and hyperparameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model
joblib.dump(best_model, 'heart_disease_model.joblib')
print("Model saved to 'heart_disease_model.joblib'")
