import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Prepare the XOR Data
# Input data (features)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output data (labels)
y = np.array([[0], [1], [1], [0]])

# Step 2: Define the DNN Model
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),  # Hidden layer with 8 neurons
    Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
model.fit(X, y, epochs=500, verbose=0)  # Train for 500 epochs

# Step 4: Evaluate and Test the Model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# Test predictions
predictions = model.predict(X)
print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Input: {X[i]}, Predicted: {pred[0]:.2f}, Rounded: {round(pred[0])}")
