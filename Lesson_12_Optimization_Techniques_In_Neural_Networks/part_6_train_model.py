optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()  # Zero the gradients
    y_pred = model(x)  # Forward pass
    loss = criterion(y_pred, y)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
