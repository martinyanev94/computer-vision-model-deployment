# Generating synthetic dataset
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X + 2 + np.random.normal(0, 0.1, (100, 1))  # Adding a bit of noise

# Normalizing our input
X = (X - np.min(X)) / (np.max(X) - np.min(X))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Initialize network parameters
nn = NeuralNetwork(input_size=1, hidden_size=5, output_size=1)
nn.train(X, y, epochs=1000, learning_rate=0.01)
