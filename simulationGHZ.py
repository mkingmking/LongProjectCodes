import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Pauli operators
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
ghz = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(ghz)

# Define 3-qubit generators (acting non-trivially on qubits 0 and 1)
generators = {
    "X⊗Y⊗I": tensor(X, Y, I),
    "Y⊗X⊗I": tensor(Y, X, I),
    "X⊗X⊗I": tensor(X, X, I),
    "Y⊗Y⊗I": tensor(Y, Y, I),
    "Z⊗Z⊗I": tensor(Z, Z, I),
}

# Depolarizing noise on one qubit in 3-qubit system
def depolarize(rho, p, target):
    ops = [np.sqrt(1 - p) * tensor(I, I, I)]
    for P in [X, Y, Z]:
        if target == 0:
            ops.append(np.sqrt(p / 3) * tensor(P, I, I))
        elif target == 1:
            ops.append(np.sqrt(p / 3) * tensor(I, P, I))
        elif target == 2:
            ops.append(np.sqrt(p / 3) * tensor(I, I, P))
    return sum(K * rho * K.dag() for K in ops)

def apply_depolarizing_all(rho, p):
    for i in range(3):
        rho = depolarize(rho, p, i)
    return rho

# QFI computation for mixed states
def qfi(rho, A):
    eigvals, eigvecs = rho.eigenstates()
    val = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            lam_i, lam_j = eigvals[i], eigvals[j]
            if lam_i + lam_j > 0:
                delta = (lam_i - lam_j)**2 / (lam_i + lam_j)
                element = complex((eigvecs[i].dag() * A * eigvecs[j]))
                val += delta * abs(element)**2
    return 4 * val

# Simulation
ps = np.linspace(0, 1, 100)
qfi_data = {}

for label, G in generators.items():
    values = []
    for p in ps:
        U = (-1j * np.pi / 4 * G).expm()
        encoded = U * rho0 * U.dag()
        noisy = apply_depolarizing_all(encoded, p)
        values.append(qfi(noisy, G))
    qfi_data[label] = values

# Plotting subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (label, values) in enumerate(qfi_data.items()):
    axs[i].plot(ps, values, label=label)
    axs[i].set_title(f"Generator: {label}")
    axs[i].set_xlabel("Depolarizing strength p")
    axs[i].set_ylabel("QFI")
    axs[i].grid(True)
    axs[i].legend()

# Turn off unused subplot
for j in range(len(generators), len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()
