def train(network, training_data, target_data, epochs, learning_rate):
    for epoch in range(epochs):
        loss = calculate_loss(network, training_data, target_data)
        network.backpropagation(training_data, target_data, learning_rate)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss}')

# Example of how to use the train function
if __name__ == "__main__":
    input_size = 2
    hidden_size = 2
    output_size = 1
    network = FeedForwardNetwork(input_size, hidden_size, output_size)

    # Sample training data (XOR problem)
    training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_data = np.array([[0], [1], [1], [0]])

    train(network, training_data, target_data, epochs=1000, learning_rate=0.1)
