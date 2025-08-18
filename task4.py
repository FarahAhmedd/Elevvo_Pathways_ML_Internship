import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Load data
data = pd.read_csv('Datasets/loan_approval_dataset.csv')

# Separate features and target
X = data.drop(['loan_id', ' loan_status'], axis=1)
y = data[' loan_status'].map({' Approved': 1, ' Rejected': 0}).astype(int)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify column types
num_cols = X_train.select_dtypes(include=['int64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

# Preprocessing: numeric + categorical
numeric_transformer = IterativeImputer(random_state=0)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categories
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Pipeline: Imputer + Classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

# Parameters to search
param_grid = {
    'preprocessor__num__max_iter': [5, 10, 20],
    'preprocessor__num__initial_strategy': ['mean', 'median', 'most_frequent'],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20]
}

# Multiple scorers
scoring = {
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# GridSearchCV with multi-metric scoring
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring=scoring,
    refit='f1',
    cv=5,
    n_jobs=-1
)
print("Starting GridSearchCV...")
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("CV Precision:", grid.cv_results_['mean_test_precision'][grid.best_index_])
print("CV Recall:", grid.cv_results_['mean_test_recall'][grid.best_index_])
print("CV F1:", grid.cv_results_['mean_test_f1'][grid.best_index_])

# ✅ Use best model on test data
best_model = grid.best_estimator_
y_pred = best_model.predict(X_val)

# Test set metrics
print("\nTest Precision:", precision_score(y_val, y_pred, average='weighted'))
print("Test Recall:", recall_score(y_val, y_pred, average='weighted'))
print("Test F1:", f1_score(y_val, y_pred, average='weighted'))


