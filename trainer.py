from tensorflow.keras.optimizers import Adam


class Trainer:
    def __init__(self, model):
        self.model = model

    def compile_model(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, training_tokens, training_scores, epochs=10, batch_size=16, validation_split=0.2):
        self.model.fit(training_tokens, training_scores, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate_model(self, testing_tokens, testing_scores):
        loss, accuracy = self.model.evaluate(testing_tokens, testing_scores)
        return accuracy
