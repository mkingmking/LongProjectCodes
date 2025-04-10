import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Pauli matrices
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(bell)

# Generators to test
generators = {
    "X⊗Y": tensor(X, Y),
    "Y⊗X": tensor(Y, X),
    "X⊗X": tensor(X, X),
    "Y⊗Y": tensor(Y, Y),
    "Z⊗Z": tensor(Z, Z),
}

# Depolarizing noise on one qubit
def depolarize(rho, p, target):
    ops = [np.sqrt(1 - p) * tensor(I, I)]
    for P in [X, Y, Z]:
        if target == 0:
            ops.append(np.sqrt(p / 3) * tensor(P, I))
        else:
            ops.append(np.sqrt(p / 3) * tensor(I, P))
    return sum(K * rho * K.dag() for K in ops)

def apply_depolarizing_both(rho, p):
    rho = depolarize(rho, p, 0)
    rho = depolarize(rho, p, 1)
    return rho

# QFI calculation
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
        noisy = apply_depolarizing_both(encoded, p)
        values.append(qfi(noisy, G))
    qfi_data[label] = values

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (label, values) in enumerate(qfi_data.items()):
    axs[i].plot(ps, values, label=label)
    axs[i].set_title(f"Generator: {label}")
    axs[i].set_xlabel("Depolarizing strength p")
    axs[i].set_ylabel("QFI")
    axs[i].grid(True)
    axs[i].legend()

# Turn off unused subplot if any
for j in range(len(generators), len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()
