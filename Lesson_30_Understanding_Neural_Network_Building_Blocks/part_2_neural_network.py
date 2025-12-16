class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Neuron(np.random.rand(input_size, hidden_size), np.random.rand(hidden_size))
        self.output_layer = Neuron(np.random.rand(hidden_size, output_size), np.random.rand(output_size))

    def forward(self, inputs):
        hidden_output = self.hidden_layer.activate(inputs)
        output = self.output_layer.activate(hidden_output)
        return output
