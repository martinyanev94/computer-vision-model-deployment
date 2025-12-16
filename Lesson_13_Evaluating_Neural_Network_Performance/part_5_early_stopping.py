def early_stopping(training_losses, validation_losses, patience):
    if len(validation_losses) > patience:
        if validation_losses[-1] > min(validation_losses[-(patience + 1):-1]):
            print("Early stopping activated!")
            return True
    return False

# Placeholder for losses during training
training_losses = []
validation_losses = []

# Simulate training loop
for epoch in range(100):  # Run for 100 epochs
    output = feedforward(X, weights)
    train_loss = mean_squared_error(y_true, output)
    training_losses.append(train_loss)
    
    # Simulated validation loss (could be obtained from a separate dataset)
    val_loss = train_loss + np.random.uniform(-0.1, 0.1)  # Introducing some variability
    validation_losses.append(val_loss)
    
    backpropagation(X, y_true, output, weights, learning_rate)

    if early_stopping(training_losses, validation_losses, patience=5):
        break
