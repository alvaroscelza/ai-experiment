from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import Trainer

# Load and preprocess data
data_loader = DataLoader(file_path='reviews.csv')
texts, scores = data_loader.load_data()
training_tokens, testing_tokens, training_scores, testing_scores, tokenizer = data_loader.preprocess_data(texts, scores)

# Load GloVe embeddings and build model
model_builder = ModelBuilder(tokenizer)
embedding_matrix = model_builder.load_glove_embeddings('glove.6B.50d.txt')
model = model_builder.build_model(embedding_matrix)

# Compile, train, and evaluate the model
trainer = Trainer(model)
trainer.compile_model()
trainer.train_model(training_tokens, training_scores)
accuracy = trainer.evaluate_model(testing_tokens, testing_scores)

print(f'Test accuracy: {accuracy * 100:.2f}%')
