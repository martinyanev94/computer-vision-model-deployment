# Example Input
input_data = np.array([1, 0])  # Binary input
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Forward pass
output = nn.feedforward(input_data)
print("Output after feedforward:", output)
