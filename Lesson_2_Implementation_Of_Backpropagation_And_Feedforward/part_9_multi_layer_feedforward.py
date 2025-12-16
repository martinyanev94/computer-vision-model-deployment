# Define weights for multiple layers
weights_hidden = np.random.rand(2, 3)  # Weights from input to hidden layer (2 inputs -> 3 hidden neurons)
weights_output = np.random.rand(3, 2)  # Weights from hidden to output layer (3 hidden neurons -> 2 outputs)

# Feedforward for multi-layer network
def feedforward_multi_layer(input_data, weights_hidden, weights_output):
    hidden_layer = relu(np.dot(input_data, weights_hidden))  # Hidden layer activation
    output_layer = relu(np.dot(hidden_layer, weights_output))  # Final output layer activation
    return hidden_layer, output_layer

hidden_layer_output, final_output = feedforward_multi_layer(input_data, weights_hidden, weights_output)
print(f"Hidden Layer Output: {hidden_layer_output}")
print(f"Final Output: {final_output}")
