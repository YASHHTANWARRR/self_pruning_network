import torch

def evaluate(model, testloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            _, pred = out.max(1)

            total += y.size(0)
            correct += pred.eq(y).sum().item()

    accuracy = 100 * correct / total

    total_w, pruned = 0, 0

    for m in model.modules():
        if hasattr(m, "get_gates"):
            gates = m.get_gates()

            total_w += gates.numel()
            pruned += (gates < 0.05).sum().item()

    sparsity = 100 * pruned / total_w

    return accuracy, sparsity