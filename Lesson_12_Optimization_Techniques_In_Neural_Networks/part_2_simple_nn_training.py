import torch.nn as nn
import torch.optim as optim

# A simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.fc(x)

# Create the model, define the criterion and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Sample data
x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=False)
y = torch.tensor([[2.0], [3.0], [4.0]], requires_grad=False)

# Training loop
for epoch in range(100):
    model.train()
    
    optimizer.zero_grad()  # Zero the gradients
    y_pred = model(x)  # Forward pass
    loss = criterion(y_pred, y)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
