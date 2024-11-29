import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Generate random noise
def generate_latent_points(latent_dim, n_samples):
    return np.random.randn(latent_dim * n_samples).reshape(n_samples, latent_dim)

# Create generator model
def build_generator(latent_dim):
    model = Sequential([
        Dense(128, activation=LeakyReLU(0.2), input_dim=latent_dim),
        Dense(256, activation=LeakyReLU(0.2)),
        Dense(28 * 28, activation='tanh'),
        Reshape((28, 28))
    ])
    return model

# Build and train
latent_dim = 100
generator = build_generator(latent_dim)

# Generate images
noise = generate_latent_points(latent_dim, 16)
generated_images = generator.predict(noise)

# Visualize
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.show()
