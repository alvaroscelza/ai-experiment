import faulthandler

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from data_loader import DataLoader

# Load and preprocess data
data_loader = DataLoader(file_path='reviews.csv')
texts, scores = data_loader.load_data()
training_tokens, testing_tokens, training_scores, testing_scores, tokenizer = data_loader.preprocess_data(texts, scores)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
with joblib.parallel_backend('threading', n_jobs=2):
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=4)
    grid_search.fit(training_tokens, training_scores)

# Best parameters from GridSearchCV
print(f'Best parameters found: {grid_search.best_params_}')

# Train the model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(training_tokens, training_scores)

# Evaluate the model
train_predictions = best_model.predict(training_tokens)
test_predictions = best_model.predict(testing_tokens)

# Calculate and print accuracy
train_accuracy = accuracy_score(training_scores, train_predictions)
test_accuracy = accuracy_score(testing_scores, test_predictions)

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')
