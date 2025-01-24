import matplotlib.pyplot as plt


def plot_convergence(iterations, energies):
    """Plot energy convergence during optimization."""
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Energy (Hartree)')
    plt.title('Energy Convergence')
    plt.grid()
    plt.show()
