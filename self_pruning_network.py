import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import GradScaler, autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

#plotting function
def plot_results(results):
    lambdas = [r[0] for r in results]
    accs = [r[1] for r in results]
    sparsities = [r[2] for r in results]

    plt.figure()
    plt.plot(lambdas, accs, marker='o')
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Lambda")
    plt.savefig("accuracy_vs_lambda.png")
    plt.close()

    plt.figure()
    plt.plot(lambdas, sparsities, marker='o')
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Sparsity (%)")
    plt.title("Sparsity vs Lambda")
    plt.savefig("sparsity_vs_lambda.png")
    plt.close()

    plt.figure()
    plt.plot(sparsities, accs, marker='o')
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Sparsity Tradeoff")
    plt.savefig("accuracy_vs_sparsity.png")
    plt.close()

#nn module
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def sparsity_info(self, threshold=0.1):
        gates = torch.sigmoid(self.gate_scores)
        total = gates.numel()
        pruned = (gates < threshold).sum().item()
        return total, pruned

#self pruning module
class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            PrunableLinear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(256, 128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

#new sparsity block
    def sparsity_loss(self):
        total = 0
        for layer in self.prunable_layers():
            g = torch.sigmoid(layer.gate_scores)
            total += (g + 0.1 * g * (1 - g)).mean()
        return total
    
    def network_sparsity(self, threshold=0.2): #threshold updated
        total, pruned = 0, 0
        for layer in self.prunable_layers():
            t, p = layer.sparsity_info(threshold)
            total += t
            pruned += p
        return 100 * pruned / total

    def gate_values(self):
        vals = []
        for layer in self.prunable_layers():
            vals.append(torch.sigmoid(layer.gate_scores).detach().cpu().numpy().ravel())
        return np.concatenate(vals)


#data loading and training functions
def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

#training loop
def train_one_epoch(model, loader, optimizer, scaler, lam):
    model.train()
    total_loss = cls_loss_sum = sp_loss_sum = 0

    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=(DEVICE.type == "cuda")):
            out = model(x)
            cls = F.cross_entropy(out, y)
            sp = model.sparsity_loss()
            loss = cls + lam * sp

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        cls_loss_sum += cls.item()
        sp_loss_sum += sp.item()

    n = len(loader)
    return total_loss / n, cls_loss_sum / n, sp_loss_sum / n

#evaluation function
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100 * correct / total

#main training function
def train_model(lam):
    train_loader, test_loader = get_dataloaders()
    model = SelfPruningNet().to(DEVICE)

    gate_params = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = optim.Adam([
        {"params": other_params, "lr": 3e-3},
        {"params": gate_params, "lr": 5e-2}
    ])

    scaler = GradScaler()

    for epoch in range(1, 41):
        total, cls, sp = train_one_epoch(model, train_loader, optimizer, scaler, lam)
        
        for layer in model.prunable_layers():
            g = torch.sigmoid(layer.gate_scores)
            print("Gate stats:", g.min().item(), g.mean().item(), g.max().item())
            break
        
        acc = evaluate(model, test_loader)
        sparsity = model.network_sparsity()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:2d} loss={total:.3f} acc={acc:.2f}% sparse={sparsity:.1f}%")

    return model, acc, sparsity

#main function to run experiments
def main():
    lambdas = [1e-2, 1e-1, 1]#updated values
    results = []
    
    #updated main loop to save models and track best model
    best_acc = 0
    best_model = None
    best_lambda = None

    for lam in lambdas:
        print(f"\nTraining λ = {lam}")
        model, acc, sparsity = train_model(lam)

        results.append((lam, acc, sparsity))

        torch.save(model.state_dict(), f"model_lambda_{lam}.pth")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_lambda = lam

    torch.save(best_model.state_dict(), "best_model.pth")

    print("\nResults:")
    for lam, acc, sp in results:
        print(f"{lam} → Acc: {acc:.2f}% | Sparsity: {sp:.1f}%")

    print(f"\nBest model: λ = {best_lambda} | Acc = {best_acc:.2f}%")

    plot_results(results)

if __name__ == "__main__":
    main()