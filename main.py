from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import Trainer
from sklearn.metrics import accuracy_score

# Load and preprocess data
data_loader = DataLoader(file_path='reviews.csv')
texts, scores = data_loader.load_data()
training_tokens, testing_tokens, training_scores, testing_scores, tokenizer = data_loader.preprocess_data(texts, scores)

# Build model
model_builder = ModelBuilder(tokenizer)
model = model_builder.build_model()

# Compile, train, and evaluate the model
trainer = Trainer(model)
trainer.compile_model()
trainer.train_model(training_tokens, training_scores)

# Evaluate the model
train_predictions = model.predict(training_tokens)
test_predictions = model.predict(testing_tokens)

train_accuracy = accuracy_score(training_scores, train_predictions.argmax(axis=1))
test_accuracy = accuracy_score(testing_scores, test_predictions.argmax(axis=1))

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')
