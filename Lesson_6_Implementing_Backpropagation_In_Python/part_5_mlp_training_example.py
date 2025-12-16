# Sample data
inputs = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
y_true = np.array([0, 0, 1])  # Expected output

# Initialize MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1)
learning_rate = 0.01

# Training loop
for epoch in range(1000):
    for x in inputs:
        # Forward pass
        pred = mlp.forward(x)
        # Backward pass
        mlp.backward(np.array([0]), learning_rate)

# Testing predictions after training
print("Final predictions after training:")
for x in inputs:
    print(mlp.forward(x))
