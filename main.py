from train import train
from evaluate import evaluate
from plot import plot_gate_distribution

#first iteration lambdas
#lambdas = [1e-4, 1e-3, 1e-2]

#second iteration lambdas
lambdas = [1e-3, 1e-2, 1e-1]

results = []

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")

    model, testloader = train(lambda_val=lam, epochs=5)

    acc, sparsity = evaluate(model, testloader)

    print(f"Lambda: {lam}, Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")

    results.append((lam, acc, sparsity))

# Plot last model
plot_gate_distribution(model)