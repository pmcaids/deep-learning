import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load and Preprocess Data
# Using a more comprehensive input text (a short story excerpt)
text = """
In the heart of an ancient forest, where shadows danced between towering trees, 
a young traveler named Aria wandered. Her journey had been long and treacherous, 
guided only by a tattered map and an unwavering spirit. The wind whispered secrets 
of forgotten paths, and the moonlight cast ethereal patterns on the moss-covered ground.

Aria's quest was simple yet profound: to find the legendary Crystal of Wisdom, 
a mystical artifact said to grant profound insights to those pure of heart. 
Each step brought her closer to her destiny, though the path was fraught with 
unknown dangers and mystical challenges.
"""

# Preprocessing
chars = sorted(list(set(text)))  # Get unique characters
char_to_index = {c: i for i, c in enumerate(chars)}  # Map characters to indices
index_to_char = {i: c for i, c in enumerate(chars)}  # Map indices to characters

# Prepare sequences
seq_length = 40  # Increased sequence length
step = 3  # Reduced step size for more training data
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

# Convert sequences to numerical format
X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)

for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Step 2: Build an Enhanced RNN Model
model = Sequential([
    LSTM(256, input_shape=(seq_length, len(chars)), return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Step 3: Train the Model
history = model.fit(X, y, batch_size=64, epochs=50, callbacks=[early_stopping])

# Step 4: Enhanced Text Generation
def generate_text(seed, length, temperature=1.0):
    generated_text = seed
    current_seq = seed
    
    for _ in range(length):
        # Prepare input sequence
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(current_seq):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1
        
        # Predict next character with temperature
        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Sample from the probability distribution
        next_index = np.random.choice(len(chars), p=preds)
        next_char = index_to_char[next_index]
        
        # Update generated text
        generated_text += next_char
        current_seq = current_seq[1:] + next_char
    
    return generated_text

# Demonstrate text generation with different seeds
print("Seed: 'In the '")
generated1 = generate_text("In the ", length=200, temperature=0.7)
print(generated1)

print("\n\nSeed: 'Aria walk'")
generated2 = generate_text("Aria walk", length=250, temperature=0.5)
print(generated2)

# Optional: Visualize training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
