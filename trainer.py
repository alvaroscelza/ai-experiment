from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Trainer:
    def __init__(self, model, learning_rate=0.0001, batch_size=32, epochs=30):  # Adjusted hyperparameters
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def compile_model(self):
        custom_optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, training_tokens, training_scores):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        self.model.fit(training_tokens, training_scores, epochs=self.epochs, validation_split=0.2,
                       batch_size=self.batch_size, callbacks=[early_stopping, reduce_lr])

    def evaluate_model(self, testing_tokens, testing_scores):
        loss, accuracy = self.model.evaluate(testing_tokens, testing_scores)
        return accuracy
