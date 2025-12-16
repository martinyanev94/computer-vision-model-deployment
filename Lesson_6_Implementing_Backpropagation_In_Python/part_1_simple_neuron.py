class SimpleNeuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.a = self.sigmoid(self.z)
        return self.a
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, y_true, learning_rate):
        # Calculate loss and gradient w.r.t output activation
        m = y_true.size
        dz = (self.a - y_true) / m  # Gradient of cost function w.r.t activation
        dw = np.dot(dz, self.inputs)  # Gradient of activation w.r.t weights
        db = np.sum(dz)  # Gradient of activation w.r.t bias
        
        # Update weights and bias
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
