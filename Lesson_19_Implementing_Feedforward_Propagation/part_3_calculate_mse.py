expected_output = np.array([[0]])
error = expected_output - output
mse = np.mean(error ** 2)
