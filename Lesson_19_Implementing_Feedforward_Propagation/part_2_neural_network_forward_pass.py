# Input values
inputs = np.array([[1], [1]])

# Calculate hidden layer inputs
hidden_layer_input = np.dot(weights_input_hidden.T, inputs)
hidden_layer_output = sigmoid(hidden_layer_input)
# Calculate output layer inputs
output_layer_input = np.dot(weights_hidden_output.T, hidden_layer_output)
output = sigmoid(output_layer_input)
