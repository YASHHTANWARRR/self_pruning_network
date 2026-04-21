import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

from model import PrunableNet
from utils import compute_sparsity_loss


def get_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def get_dataloaders():
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return trainloader, testloader


def train(lambda_val=1e-3, epochs=10):

    device = get_device()
    trainloader, testloader = get_dataloaders()

    model = PrunableNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in trainloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                out = model(x)

                cls_loss = F.cross_entropy(out, y)
                sparsity_loss = compute_sparsity_loss(model)

                loss = cls_loss + lambda_val * sparsity_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: {total_loss:.4f}")

    return model, testloader