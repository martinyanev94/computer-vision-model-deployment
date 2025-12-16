import matplotlib.pyplot as plt

training_losses = []
validation_losses = []

# Simulate training over epochs
for epoch in range(100):
    output = feedforward(X, weights)
    train_loss = mean_squared_error(y_true, output)
    training_losses.append(train_loss)

    # Simulated validation loss
    val_loss = train_loss + np.random.uniform(-0.1, 0.1)
    validation_losses.append(val_loss)
    
    backpropagation(X, y_true, output, weights, learning_rate)

# Plotting the training and validation losses
plt.plot(training_losses, label='Training Loss', color='blue')
plt.plot(validation_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()
