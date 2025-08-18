import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np

print("Importing libraries...")
# Load data
data = pd.read_csv('Datasets/loan_approval_dataset.csv')
print("Loading datasets...")


# Separate features and target
X = data.drop(['loan_id', ' loan_status'], axis=1)
y = data[' loan_status'].map({' Approved': 1, ' Rejected': 0})
print("Preparing data and splitting...")

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Identify column types
num_cols = X_train.select_dtypes(include=['int64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

# Define imputers to test
imputers = {
    'IterativeImputer': IterativeImputer(random_state=0),
    'SimpleImputer': SimpleImputer(strategy='mean')
}

# Classifiers and their param grids
classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }),
    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {
        'clf__C': [0.1, 1, 10]
    }),
    'DecisionTree': (DecisionTreeClassifier(random_state=42), {
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5, 10]
    }),
}

# Scorers
scoring = {
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

results = []

for imputer_name, imputer in imputers.items():
    print(f"\n=== Using {imputer_name} for numeric data ===")
    
    numeric_transformer = imputer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    for use_smote in [False, True]:
        print(f"\n-- SMOTE: {'Enabled' if use_smote else 'Disabled'} --")
        
        # Pipeline with or without SMOTE
        if use_smote:
            pipe_steps = [
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier())  # placeholder
            ]
            PipelineClass = ImbPipeline
        else:
            pipe_steps = [
                ('preprocessor', preprocessor),
                ('clf', RandomForestClassifier())  # placeholder
            ]
            PipelineClass = Pipeline
        
        for clf_name, (clf, param_grid) in classifiers.items():
            print(f"Classifier: {clf_name}")
            
            # Create pipeline and param grid for GridSearch
            pipeline = PipelineClass(pipe_steps)
            
            # Override placeholder classifier and add its param grid
            pipeline.set_params(clf=clf)
            
            # Combine parameters with imputer param grid only if IterativeImputer
            full_param_grid = param_grid.copy()
            
            if imputer_name == 'IterativeImputer':
                # Add IterativeImputer params to grid search
                full_param_grid.update({
                    'preprocessor__num__max_iter': [5, 10, 20],
                    'preprocessor__num__initial_strategy': ['mean', 'median', 'most_frequent']
                })
            
            # Grid search
            grid = GridSearchCV(
                pipeline,
                param_grid=full_param_grid,
                scoring=scoring,
                refit='f1',
                cv=5,
                n_jobs=-1
            )
            
            grid.fit(X_train, y_train)
            
            best_clf = type(grid.best_estimator_.named_steps['clf']).__name__
            best_params = grid.best_params_
            best_idx = grid.best_index_
            
            # Predict on validation
            y_pred = grid.predict(X_val)
            
            # Collect results
            res = {
                'Imputer': imputer_name,
                'SMOTE': use_smote,
                'Classifier': clf_name,
                'Best Params': best_params,
                'CV Precision': grid.cv_results_['mean_test_precision'][best_idx],
                'CV Recall': grid.cv_results_['mean_test_recall'][best_idx],
                'CV F1': grid.cv_results_['mean_test_f1'][best_idx],
                'Test Precision': precision_score(y_val, y_pred, average='weighted'),
                'Test Recall': recall_score(y_val, y_pred, average='weighted'),
                'Test F1': f1_score(y_val, y_pred, average='weighted'),
            }
            results.append(res)
            
            print(f"  Best CV F1: {res['CV F1']:.4f} | Test F1: {res['Test F1']:.4f}")

# Summarize all results in a DataFrame
results_df = pd.DataFrame(results)
print("\n\n=== Summary of all experiments ===")
print(results_df[['Imputer', 'SMOTE', 'Classifier', 'CV Precision', 'CV Recall', 'CV F1', 'Test Precision', 'Test Recall', 'Test F1']])
