from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from data_loader import DataLoader

# Load and preprocess data
data_loader = DataLoader(file_path='reviews.csv')
texts, scores = data_loader.load_data()
training_tokens, testing_tokens, training_scores, testing_scores, tokenizer = data_loader.preprocess_data(texts, scores)

# Initialize and train the model
model = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, random_state=42)
training_scores -= 1  # XGBoost expects class labels to start from 0. Since our labels start from 1, we subtract 1.
testing_scores -= 1  # Subtract 1 from testing scores as well
model.fit(training_tokens, training_scores)

# Evaluate the model
train_predictions = model.predict(training_tokens)
test_predictions = model.predict(testing_tokens)

# Calculate and print accuracy
train_accuracy = accuracy_score(training_scores, train_predictions)
test_accuracy = accuracy_score(testing_scores, test_predictions)

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')
