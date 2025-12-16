# Example weights and bias for output layer
weights_hidden_output = np.array([0.5, 0.6, 0.8])  # Weights from hidden to output
bias_output = 0.4  # Bias for output layer

# Calculate output layer activation
output_layer_input = np.dot(weights_hidden_output, hidden_layer_output) + bias_output
output_layer_output = sigmoid(output_layer_input)  # Can also use softmax for multi-class

print("Output Layer Output:", output_layer_output)
