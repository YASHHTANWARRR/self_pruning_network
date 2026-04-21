import matplotlib.pyplot as plt

def plot_gate_distribution(model):
    all_gates = []

    for m in model.modules():
        if hasattr(m, "get_gates"):
            gates = m.get_gates().detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()