def train(model, criterion, optimizer, data_loader, epochs):
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
