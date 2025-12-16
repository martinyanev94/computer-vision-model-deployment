def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def calculate_loss(network, input_data, target_data):
    predictions = network.feedforward(input_data)
    loss = mean_squared_error(predictions, target_data)
    return loss
