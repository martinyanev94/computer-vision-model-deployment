import numpy as np

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Weights initialization
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets, epochs=10000):
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(inputs, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)

            # Calculate error
            error = targets - final_output

            # Backpropagation
            d_final_output = error * self.sigmoid_derivative(final_output)
            error_hidden_layer = d_final_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_output)

            # Update weights
            self.weights_hidden_output += hidden_output.T.dot(d_final_output) * self.learning_rate
            self.weights_input_hidden += inputs.T.dot(d_hidden_layer) * self.learning_rate

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

# Example usage
if __name__ == "__main__":
    ann = SimpleANN(input_size=3, hidden_size=4, output_size=1)
    training_data = np.array([[0, 0, 1],
                               [1, 0, 1],
                               [0, 1, 1],
                               [1, 1, 1]])
    target_data = np.array([[0], [1], [1], [0]])
    
    ann.train(training_data, target_data)
