import numpy as np

# Define the activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Define the forward propagation function
def feedforward(X, weights):
    # Multiply input by weights for each layer
    layer1_output = relu(np.dot(X, weights[0]))  # First hidden layer
    layer2_output = relu(np.dot(layer1_output, weights[1]))  # Second hidden layer
    final_output = np.dot(layer2_output, weights[2])  # Output layer
    return final_output

# Example input (2D array of shape (num_samples, num_features))
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
weights = [np.random.rand(2, 3), np.random.rand(3, 3), np.random.rand(3, 1)]  # Random weights for 3 layers

output = feedforward(X, weights)
print("Feedforward output:\n", output)
