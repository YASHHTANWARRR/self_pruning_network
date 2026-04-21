def compute_sparsity_loss(model):
    loss = 0.0
    for gates in model.get_all_gates():
        loss += gates.mean()
        loss += 0.1 * (gates * (1 - gates)).mean()
    return loss