import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------
# Custom Prunable Linear Layer
# -----------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate parameters
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(5 * self.gate_scores)
        pruned_weights = self.weight * gates
        return torch.matmul(x, pruned_weights.t()) + self.bias


# -----------------------------
# Neural Network
# -----------------------------
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 256)
        self.fc2 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Sparsity Loss
# -----------------------------
def sparsity_loss(model):
    loss = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
            total_params += gates.numel()

    return loss / total_params


# -----------------------------
# Training
# -----------------------------
def train(model, loader, optimizer, criterion, lambda_val):
    model.train()
    total_loss = 0

    for data, target in loader:
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss += lambda_val * sparsity_loss(model)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    return correct / total


# -----------------------------
# Sparsity Calculation
# -----------------------------
def calculate_sparsity(model, threshold=0.1):
    total = 0
    zero = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(5 * module.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return zero / total


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    lambdas = [0.001, 0.01, 0.05]

    acc_list = []
    sparsity_list = []
    lambda_list = []

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")

        model = PrunableNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            loss = train(model, train_loader, optimizer, criterion, lam)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        acc = evaluate(model, test_loader)
        sparsity = calculate_sparsity(model)

        print(f"Accuracy: {acc:.4f}")
        print(f"Sparsity: {sparsity:.4f}")

        acc_list.append(acc)
        sparsity_list.append(sparsity)
        lambda_list.append(lam)

    # -----------------------------
    # PLOTS
    # -----------------------------
    plt.figure(figsize=(10, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(lambda_list, acc_list, marker='o')
    plt.title("Accuracy vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")

    # Sparsity plot
    plt.subplot(1, 2, 2)
    plt.plot(lambda_list, sparsity_list, marker='o')
    plt.title("Sparsity vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Sparsity")

    plt.tight_layout()
    plt.show()
