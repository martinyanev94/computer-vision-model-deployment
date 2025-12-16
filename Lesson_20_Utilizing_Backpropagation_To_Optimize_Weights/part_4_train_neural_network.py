def train_neural_network(inputs, expected_output, weights_input_hidden, weights_hidden_output, learning_rate):
    # Feedforward pass
    hidden_layer_input = np.dot(weights_input_hidden.T, inputs)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(weights_hidden_output.T, hidden_layer_output)
    predicted_output = sigmoid(output_layer_input)

    # Calculate error
    error = expected_output - predicted_output
    
    # Backpropagation
    output_gradient = sigmoid_derivative(predicted_output) * error
    weights_hidden_output += hidden_layer_output.T @ output_gradient * learning_rate
    
    hidden_layer_error = output_gradient @ weights_hidden_output.T
    hidden_layer_gradient = sigmoid_derivative(hidden_layer_output) * hidden_layer_error
    weights_input_hidden += inputs @ hidden_layer_gradient * learning_rate

    return predicted_output, weights_input_hidden, weights_hidden_output
