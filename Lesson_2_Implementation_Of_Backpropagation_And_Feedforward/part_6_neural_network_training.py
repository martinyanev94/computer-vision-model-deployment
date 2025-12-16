# Assume we have an output from the previous example
output = feedforward(input_data, weights, bias)
target = np.array([0.0, 1.0])  # Our true labels

# Define a simple loss function
def loss_function(output, target):
    return np.mean((output - target) ** 2)

# Calculate the loss
loss = loss_function(output, target)
print(f"Loss: {loss}")

# Backpropagation
def backpropagate(input_data, output, target, weights, learning_rate=0.01):
    # Calculate the derivative of the loss
    error = output - target  # Compute the error
    d_loss = error * (1 * (output > 0))  # Derivative of ReLU is 1 for positive input

    # Calculate gradients
    d_weights = np.dot(input_data.reshape(-1, 1), d_loss.reshape(1, -1))
    
    # Update weights and bias
    weights -= learning_rate * d_weights
    return weights

# Update weights using backpropagation
weights = backpropagate(input_data, output, target, weights)
print(f"Updated Weights: {weights}")
