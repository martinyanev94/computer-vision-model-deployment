class SimpleNNWithDropout(nn.Module):
    def __init__(self):
        super(SimpleNNWithDropout, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with a 50% probability
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

model = SimpleNNWithDropout()  # Instantiate the model with dropout
