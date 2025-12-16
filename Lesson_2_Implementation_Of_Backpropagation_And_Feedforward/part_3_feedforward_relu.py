import numpy as np

# Define the activation function
def relu(x):
    return np.maximum(0, x)

# Define the weights and biases
weights = np.array([[0.2, 0.8], [0.5, 0.3]])
bias = np.array([0.1, 0.1])

# Input data
input_data = np.array([1, 0.5])

# Feedforward calculation
def feedforward(input_data, weights, bias):
    z = np.dot(input_data, weights) + bias  # Linear combination
    return relu(z)  # Apply activation function

output = feedforward(input_data, weights, bias)
print(f"Output of feedforward: {output}")
