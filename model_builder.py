import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = tf.nn.tanh(tf.tensordot(x, self.W, axes=(2, 0)) + self.b)
        at = tf.nn.softmax(et, axis=1)
        output = tf.reduce_sum(x * at, axis=1)
        return output

class ModelBuilder:
    def __init__(self, tokenizer, embedding_dim=300, max_sequence_length=100, num_classes=5):
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
        model.add(Bidirectional(LSTM(200, return_sequences=True)))
        model.add(Dropout(0.3))  # Increased dropout rate
        model.add(Bidirectional(LSTM(200, return_sequences=True)))
        model.add(Attention())  # Attention layer
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.0001)))
        return model
