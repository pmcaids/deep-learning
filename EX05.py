import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Step 1: Load and Preprocess Data
vocab_size = 10000  # Number of unique words to consider
max_length = 100  # Maximum length of a review
embedding_dim = 128

# Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform length
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

# Step 2: Build the LSTM Model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),  # Embedding layer
    LSTM(128, return_sequences=False),  # LSTM layer
    Dropout(0.5),  # Dropout for regularization
    Dense(128, activation='relu'),  # Fully connected layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Step 4: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Step 5: Test with a Custom Review
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

sample_review = X_test[0]
decoded_review = decode_review(sample_review)
print(f"Decoded Review: {decoded_review}")

prediction = model.predict(sample_review.reshape(1, -1))
print(f"Sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")
