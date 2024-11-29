import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Prepare Dataset
def load_data(data_dir, img_size=(64, 64)):
    X, y = [], []
    label_map = {}
    current_label = 0
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        if person_name not in label_map:
            label_map[person_name] = current_label
            current_label += 1
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label_map[person_name])
    
    return np.array(X), np.array(y), label_map

# Load your dataset
data_dir = "E:\DEEP LEARNING\DATASET"  # Replace with your dataset path
X, y, label_map = load_data(data_dir)
X = X / 255.0  # Normalize pixel values to [0, 1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')  # Output layer for classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 3: Train the Model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Step 4: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Step 5: Visualize Results
def visualize_prediction(index):
    img = X_test[index]
    true_label = y_test[index]
    predicted_label = np.argmax(model.predict(img[np.newaxis, ...]))
    
    plt.imshow(img)
    plt.title(f"True: {list(label_map.keys())[true_label]}, Predicted: {list(label_map.keys())[predicted_label]}")
    plt.axis('off')
    plt.show()

# Show a prediction
visualize_prediction(0)
