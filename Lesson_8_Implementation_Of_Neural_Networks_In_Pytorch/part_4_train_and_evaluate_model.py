num_epochs = 5  # Number of epochs to train

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
def evaluate_model(test_loader):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disables gradient tracking
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

evaluate_model(test_loader)
