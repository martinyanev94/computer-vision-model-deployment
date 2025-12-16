def accuracy(network, test_data, test_labels):
    predictions = network.feedforward(test_data)
    predictions = np.round(predictions)  # Round the predictions to get binary outputs
    correct_predictions = np.sum(predictions == test_labels)
    return correct_predictions / len(test_labels)

# Evaluating Model Performance
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_labels = np.array([[0], [1], [1], [0]])
print("Accuracy:", accuracy(network, test_data, test_labels))
