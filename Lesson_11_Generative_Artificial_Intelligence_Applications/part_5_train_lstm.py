def train_lstm(model, data_loader, epochs, criterion, optimizer):
    for epoch in range(epochs):
        for sequences, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
