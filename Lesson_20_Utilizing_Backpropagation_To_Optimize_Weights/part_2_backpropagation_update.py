weights_hidden_output += hidden_layer_output.T @ output_gradient
hidden_layer_error = output_gradient @ weights_hidden_output.T
hidden_layer_gradient = sigmoid_derivative(hidden_layer_output) * hidden_layer_error
