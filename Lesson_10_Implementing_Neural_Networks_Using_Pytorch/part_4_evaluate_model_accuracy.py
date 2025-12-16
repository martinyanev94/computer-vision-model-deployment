# Evaluating the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean().item()

print(f'Accuracy of the model on the test set: {accuracy:.4f}')
