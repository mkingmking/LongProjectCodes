import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Pauli operators
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state: (|000> + |111>)/sqrt(2)
ghz = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) + 
       tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(ghz)

# 3-body generators
generators = {
    "X⊗Y⊗Z": tensor(X, Y, Z),
    "Y⊗X⊗Z": tensor(Y, X, Z),
    "X⊗X⊗X": tensor(X, X, X),
    "Y⊗Y⊗Y": tensor(Y, Y, Y),
    "Z⊗Z⊗Z": tensor(Z, Z, Z),
}

# Depolarizing noise on one qubit
def depolarize(rho, p, target):
    ops = [np.sqrt(1 - p) * tensor(I, I, I)]
    for P in (X, Y, Z):
        if target == 0:
            ops.append(np.sqrt(p / 3) * tensor(P, I, I))
        elif target == 1:
            ops.append(np.sqrt(p / 3) * tensor(I, P, I))
        else:
            ops.append(np.sqrt(p / 3) * tensor(I, I, P))
    return sum(K * rho * K.dag() for K in ops)

# Apply depolarizing to all qubits
def apply_depolarizing_all(rho, p):
    for i in range(3):
        rho = depolarize(rho, p, i)
    return rho

# QFI computation: sum only i<j, standard factor for mixed states
def qfi(rho, A):
    eigvals, eigvecs = rho.eigenstates()
    q = 0
    N = len(eigvals)
    for i in range(N):
        for j in range(i):
            lam_i, lam_j = eigvals[i], eigvals[j]
            if lam_i + lam_j > 0:
                delta = (lam_i - lam_j)**2 / (lam_i + lam_j)
                element = eigvecs[i].dag() * A * eigvecs[j]
                q += delta * abs(element)**2
    # Use F = 4 * q for pure-state consistency
    return 4 * q

# Simulation over depolarizing strength p
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
