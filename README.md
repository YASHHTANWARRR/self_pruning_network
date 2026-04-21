# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training using a learnable gating mechanism.

The model is trained on the **CIFAR-10 dataset** and demonstrates how sparsity can be induced without manual pruning, while maintaining high accuracy.

---

## Key Features

* Learnable **gating mechanism** for weight pruning
* End-to-end **differentiable sparsity optimization**
* Trade-off control via regularization parameter (λ)
* High compression with minimal accuracy drop
* Automatic **graph generation** and **model saving**

---

## Methodology

### Prunable Layer

Each linear layer is augmented with learnable gates:

```
W_pruned = W * sigmoid(gate_scores)
```

### Sparsity Loss

The model uses a combination of:

* L1-style penalty on gates
* Binarization term to push gates toward 0 or 1

This encourages weights to be either fully active or completely pruned.

### Optimization Strategy

* Separate learning rates:

  * Weights: stable learning
  * Gates: faster updates for pruning
* Loss function:

```
Loss = Classification Loss + λ × Sparsity Loss
```

---

## Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.01   | 89.08%   | 90.4%    |
| 0.1    | 89.13%   | 95.3%    |
| 1      | 88.58%   | 99.1%    |

### Key Observations

* Increasing λ increases sparsity
* Accuracy remains stable even with >90% pruning
* The model retains strong performance even at extreme compression

---

## Visualizations

The project automatically generates:

* `accuracy_vs_lambda.png`
* `sparsity_vs_lambda.png`
* `accuracy_vs_sparsity.png`

These plots illustrate the trade-off between model performance and compression.

---

## Saved Models

The following models are saved:

* `model_lambda_0.01.pth`
* `model_lambda_0.1.pth`
* `model_lambda_1.pth`
* `best_model.pth`

---

## Installation

```bash
git clone <your-repo-link>
cd self_pruning_network
pip install -r requirements.txt
```

---

## Usage

Run training:

```bash
python self_pruning_network.py
```

The script will:

* Train models for multiple λ values
* Print accuracy and sparsity
* Save trained models
* Generate plots

---

## Dataset

The CIFAR-10 dataset is automatically downloaded during training.

---

## Project Structure

```
self_pruning_network.py
data/
models/
plots/
README.md
requirements.txt
```

---

## Future Improvements

* Extend pruning to convolutional layers
* Structured pruning (channel/filter level)
* Deploy pruned model for inference benchmarking

---

## Conclusion

This project demonstrates that **neural networks can learn to prune themselves effectively**, achieving:

* Up to **99% sparsity**
* Minimal loss in accuracy (~1%)

It highlights the potential of sparsity-aware training for efficient deep learning models.

---

## Author

Yash Tanwar
