# Sample data
inputs = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
y_true = np.array([0, 0, 1])  # Expected output

# Initialize neuron
neuron = SimpleNeuron(input_size=2)
learning_rate = 0.01

# Training loop
for epoch in range(1000):
    # Forward pass
    predictions = []
    for x in inputs:
        pred = neuron.forward(x)
        predictions.append(pred)
        
    # Backward pass
    neuron.backward(np.array(y_true), learning_rate)

# Testing predictions after training
print("Final predictions after training:")
for x in inputs:
    print(neuron.forward(x))
