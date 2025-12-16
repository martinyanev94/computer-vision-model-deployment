class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = SimpleNeuron(input_size)
        self.output_layer = SimpleNeuron(hidden_size)

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        output = self.output_layer.forward(hidden_output)
        return output
    
    def backward(self, y_true, learning_rate):
        self.output_layer.backward(y_true, learning_rate)
        hidden_output = self.hidden_layer.a
        self.hidden_layer.backward(np.dot(self.output_layer.weights.T, (self.output_layer.a - y_true)), learning_rate)
