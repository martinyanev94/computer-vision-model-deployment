# Backpropagation for multi-layer network
def backpropagate_multi_layer(input_data, hidden_layer_output, final_output, target, 
                              weights_hidden, weights_output, learning_rate=0.01):
    # Calculate the loss derivative for the output layer
    output_error = final_output - target
    d_output_loss = output_error * (1 * (final_output > 0)) 

    # Calculate gradient for weights_output
    d_weights_output = np.dot(hidden_layer_output.reshape(-1, 1), d_output_loss.reshape(1, -1))
    
    # Updating the weights for the output layer
    weights_output -= learning_rate * d_weights_output

    # Backpropagating the error to the hidden layer
    hidden_error = np.dot(d_output_loss, weights_output.T)
    d_hidden_loss = hidden_error * (1 * (hidden_layer_output > 0)) 

    # Calculate gradient for weights_hidden
    d_weights_hidden = np.dot(input_data.reshape(-1, 1), d_hidden_loss.reshape(1, -1))

    # Updating the weights for the hidden layer
    weights_hidden -= learning_rate * d_weights_hidden

    return weights_hidden, weights_output

# Update both sets of weights using backpropagation
weights_hidden, weights_output = backpropagate_multi_layer(input_data, hidden_layer_output, 
                                                            final_output, target, 
                                                            weights_hidden, weights_output)
print(f"Updated Weights for Hidden Layer: {weights_hidden}")
print(f"Updated Weights for Output Layer: {weights_output}")
