# Sample training data
input_size = 10  # Example input size
hidden_size1 = 20
hidden_size2 = 10
output_size = 2  # For binary classification
model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size)

# Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
for epoch in range(100):  # Let's say we train for 100 epochs
    optimizer.zero_grad()  # Clear gradients
    inputs = torch.rand(1, input_size)  # Dummy input
    labels = torch.tensor([1])  # Dummy label
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, labels)  # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
