import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten

# Load the MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 255.0  # Normalize the images
X_train = X_train.reshape(-1, 28 * 28)  # Flatten the images

# Define the generator
generator = Sequential([
    Dense(128, activation='relu', input_dim=100),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Create a simple generator
def generate_images(num_images):
    noise = np.random.normal(0, 1, size=[num_images, 100])  # Generate random noise
    generated_images = generator.predict(noise)
    
    return generated_images

# Generate and display some images
generated_images = generate_images(10)
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.show()
