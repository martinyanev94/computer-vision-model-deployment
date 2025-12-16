import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example architecture: 2-inputs, 3-hidden neurons, 1-output
input_layer = np.array([0.5, 0.2])  # Example input
weights_input_hidden = np.array([[0.3, 0.5], [0.4, 0.1], [0.2, 0.8]])  # Weights to hidden layer
bias_hidden = np.array([0.1, 0.2, 0.3])  # Biases for hidden layer

# Calculate hidden layer activations
hidden_layer_input = np.dot(weights_input_hidden, input_layer) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

print("Hidden Layer Output:", hidden_layer_output)
