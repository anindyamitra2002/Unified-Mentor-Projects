# mobile_price_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv('dataset.csv')  # adjust path if needed

# 2. Split features and target
y = df['price_range']
X = df.drop('price_range', axis=1)

# 3. Train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Build a pipeline: imputation → scaling → classifier
pipeline = Pipeline([
    ('imputer',   SimpleImputer(strategy='mean')),
    ('scaler',    StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Hyperparameter grid
param_grid = {
    'classifier__n_estimators':    [100, 200],
    'classifier__max_depth':       [None, 10, 20],
    'classifier__min_samples_split':[2, 5],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# 6. Train
grid_search.fit(X_train, y_train)

# 7. Best model & evaluation
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Save the trained model
joblib.dump(best_model, 'mobile_price_model.joblib')
print("Saved model to 'mobile_price_model.joblib'")
