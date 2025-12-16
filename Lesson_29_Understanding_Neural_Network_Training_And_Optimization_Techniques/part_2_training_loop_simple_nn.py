import torch.optim as optim

# Sample dataset
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Training loop
for epoch in range(100):  # Perform 100 epochs
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear the gradients
    
    # Forward pass
    outputs = model(inputs)  # Make predictions
    loss = criterion(outputs, targets)  # Calculate the loss
    
    # Backward pass
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
   
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
