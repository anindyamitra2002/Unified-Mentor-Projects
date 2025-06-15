# vehicle_price_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib


df = pd.read_csv('dataset.csv')
df.dropna(inplace=True)

# Target and features
y = df['price']
X = df.drop(['price'], axis=1)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing pipelines with imputation to handle any remaining NaNs
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'  # any other columns are dropped
)

# Build the full pipeline
def build_pipeline():
    rf = RandomForestRegressor(random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])
    return pipeline

pipeline = build_pipeline()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter grid for RandomForest
grid_param = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=grid_param,
    cv=5,
    n_jobs=-1,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    error_score='raise'  # will raise if any fit fails
)

# Train the pipeline with grid search
grid_search.fit(X_train, y_train)

# Retrieve the best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Evaluate on test set
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R^2: {r2:.2f}")

# Save the best model
joblib.dump(best_model, 'vehicle_price_model.joblib')
print("Model saved to 'vehicle_price_model.joblib'")

