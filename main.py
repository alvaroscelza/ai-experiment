import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2

# region load data
reviews_df = pd.read_csv('reviews.csv')
# endregion


# region Data cleaning
def combine_texts(title, text):
    if pd.isna(title) and pd.isna(text):
        return ''
    elif pd.isna(text) or text.strip() == '':
        return title
    elif pd.isna(title) or title.strip() == '':
        return text
    else:
        return f"{title} {text}"


# Combine translation_title and translation_text, handling empty translation_text
texts = reviews_df.apply(lambda row: combine_texts(row['translation_title'], row['translation_text']), axis=1)
scores = reviews_df["score"]

# Handle empty texts
combined_reviews = pd.DataFrame({'combined_text': texts, 'score': scores})
combined_reviews = combined_reviews[combined_reviews['combined_text'] != '']

# Group by 'combined_text' and average the scores, then apply ceiling
combined_reviews = combined_reviews.groupby('combined_text', as_index=False).agg({'score': 'mean'})
combined_reviews['score'] = combined_reviews['score'].apply(np.ceil)

texts, score = combined_reviews['combined_text'], combined_reviews['score']
# endregion

# region split and process data
label_encoder = LabelEncoder()
scores = label_encoder.fit_transform(scores)
training_texts, testing_texts, training_scores, testing_scores = train_test_split(texts, scores, test_size=0.2,
                                                                                  random_state=42)
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(training_texts)
training_tokens = tokenizer.texts_to_sequences(training_texts)
testing_tokens = tokenizer.texts_to_sequences(testing_texts)
training_tokens = pad_sequences(training_tokens, maxlen=100)
testing_tokens = pad_sequences(testing_tokens, maxlen=100)
# endregion

# region build model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=300,
                    trainable=True))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax', kernel_regularizer=l2(0.0001)))
# endregion

# region compile model
custom_optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# endregion

# region train model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
history = model.fit(training_tokens, training_scores, epochs=30, validation_split=0.2, batch_size=32,
                    callbacks=[early_stopping, reduce_lr])
# endregion

# region Evaluate the model
train_predictions = model.predict(training_tokens)
test_predictions = model.predict(testing_tokens)

train_accuracy = accuracy_score(training_scores, train_predictions.argmax(axis=1))
test_accuracy = accuracy_score(testing_scores, test_predictions.argmax(axis=1))

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')
# endregion
