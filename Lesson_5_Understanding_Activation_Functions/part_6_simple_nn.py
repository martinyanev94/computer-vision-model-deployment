class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden = np.random.randn(hidden_size, input_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.weights_output = np.random.randn(output_size, hidden_size)
        self.bias_output = np.random.randn(output_size)

    def forward(self, inputs):
        hidden_layer = relu(np.dot(self.weights_hidden, inputs) + self.bias_hidden)
        output_layer = sigmoid(np.dot(self.weights_output, hidden_layer) + self.bias_output)
        return output_layer

# Example usage
input_size = 2
hidden_size = 3
output_size = 1
nn = SimpleNN(input_size, hidden_size, output_size)
input_data = np.array([2, 3])
output = nn.forward(input_data)

print("Final output from the neural network:", output)
