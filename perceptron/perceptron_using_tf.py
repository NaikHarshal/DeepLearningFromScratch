import torch
import torch.nn as nn
import torch.optim as optim

# Define the perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

# Create an instance of the perceptron model
model = Perceptron(input_size=2)  # Assuming your input data has 2 features

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Sample data (replace with your actual data)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([0, 1, 1, 1], dtype=torch.float32)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()         # Calculate gradients
    optimizer.step()        # Update weights

# Make predictions
predictions = (model(X) > 0.5).float()  # Thresholding for binary classification

print(f"Predictions: {predictions}")