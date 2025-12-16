inputs = np.array([[1], [1]])
expected_output = np.array([[0]])
np.random.seed(1)

weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)

for epoch in range(10000):
    predicted_output, weights_input_hidden, weights_hidden_output = train_neural_network(
        inputs, expected_output, weights_input_hidden, weights_hidden_output, learning_rate=0.01
    )
    
print(f"Final predicted output: {predicted_output}")
