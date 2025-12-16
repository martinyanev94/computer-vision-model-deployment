# Actual label for our binary classification
y_true = np.array([1])  # Let's say the correct output is '1' (cat)

# Calculate the loss
loss = binary_cross_entropy(y_true, output)
print("Loss:", loss)
