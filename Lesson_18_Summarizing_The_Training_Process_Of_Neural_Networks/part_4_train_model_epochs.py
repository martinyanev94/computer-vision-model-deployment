# Simulating training the model for 100 epochs
for epoch in range(100):
    nn.forward(input_data)
    nn.backpropagation(input_data, y_true)
    if epoch % 10 == 0:  # Print loss every 10 epochs
        mse = mean_squared_error(y_true, nn.final_output)
        print(f"Epoch {epoch}, Mean Squared Error: {mse}")
