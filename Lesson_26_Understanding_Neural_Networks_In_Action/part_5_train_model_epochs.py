epochs = 5

for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()  # Clear gradients before each iteration
        output = model(data.view(data.size(0), -1))  # Flatten the data
        loss = criterion(output, target)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
