import numpy as np

def relu(x):
    return np.maximum(0, x)

def feedforward(X, weights):
    layer1_output = relu(np.dot(X, weights[0]))  # Process through first layer
    layer2_output = relu(np.dot(layer1_output, weights[1]))  # Process through second layer
    final_output = np.dot(layer2_output, weights[2])  # Produce final output
    return final_output

# Example input array (2 samples, 2 features)
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# Random weights for demonstration (3 layers)
weights = [np.random.rand(2, 3), np.random.rand(3, 3), np.random.rand(3, 1)]  

output = feedforward(X, weights)
print("Feedforward Output:\n", output)
