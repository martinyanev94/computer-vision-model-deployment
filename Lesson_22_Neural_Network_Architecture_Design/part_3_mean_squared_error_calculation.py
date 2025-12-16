actual_output = np.array([1])  # Assume we expected an output of 1
loss = np.mean((output_layer_output - actual_output) ** 2)
print("Mean Squared Error Loss:", loss)
