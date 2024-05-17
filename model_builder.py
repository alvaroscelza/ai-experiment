import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.regularizers import l2


class ModelBuilder:
    def __init__(self, tokenizer, embedding_dim=100, max_sequence_length=100, num_classes=5):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes

    def load_glove_embeddings(self, glove_file_path):
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        num_words = min(len(self.tokenizer.word_index) + 1, self.tokenizer.num_words)
        embedding_matrix = np.zeros((num_words, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            if i >= self.tokenizer.num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def build_model(self, embedding_matrix):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=self.embedding_dim,
                            weights=[embedding_matrix], input_length=self.max_sequence_length, trainable=True))
        model.add(Bidirectional(LSTM(200)))
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.0001)))
        return model
