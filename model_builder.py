from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


class ModelBuilder:
    def __init__(self, tokenizer, embedding_dim=300, max_sequence_length=100, num_classes=5):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=self.embedding_dim,
                            input_length=self.max_sequence_length, trainable=True))
        model.add(Bidirectional(LSTM(200, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(200)))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.0001)))
        return model
