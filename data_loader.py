import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        reviews = pd.read_csv(self.file_path)
        reviews["score"] = reviews["score"].apply(np.ceil)
        reviews['combined_text'] = reviews.apply(lambda row: f"{row['translation_title']} {row['translation_text']}",
                                                 axis=1)
        reviews.drop(columns=['translation_title', 'translation_text'], inplace=True)
        return reviews['combined_text'], reviews['score']

    @staticmethod
    def preprocess_data(texts, scores):
        training_texts, testing_texts, training_scores, testing_scores = train_test_split(texts, scores,
                                                                                          test_size=0.2,
                                                                                          random_state=42)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(training_texts)
        training_tokens = tokenizer.texts_to_sequences(training_texts)
        testing_tokens = tokenizer.texts_to_sequences(testing_texts)
        padding_length = max([len(sequence) for sequence in training_tokens])
        training_tokens = pad_sequences(training_tokens, maxlen=padding_length)
        testing_tokens = pad_sequences(testing_tokens, maxlen=padding_length)
        return training_tokens, testing_tokens, training_scores, testing_scores, tokenizer
