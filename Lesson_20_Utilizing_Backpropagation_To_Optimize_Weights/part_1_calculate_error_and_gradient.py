error = expected_output - predicted_output
output_gradient = sigmoid_derivative(predicted_output) * error
