import matplotlib.pyplot as plt

# Making predictions
predictions = nn.feedforward(X)

# Plotting the results
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, predictions, color='red', linewidth=2, label='Fitted line')
plt.legend()
plt.show()
