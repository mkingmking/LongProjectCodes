import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define Pauli matrices and identity
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# Bell state |Φ+⟩ = (|00⟩ + |11⟩) / sqrt(2)
bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(bell)

# Define tensor product generators
generators = {
    "X⊗Y": tensor(X, Y),
    "Y⊗X": tensor(Y, X),
    "X⊗X": tensor(X, X),
    "Y⊗Y": tensor(Y, Y),
    "Z⊗Z": tensor(Z, Z),
}

# Depolarizing noise (applied to each qubit independently)
def depolarize(rho, p, target):
    ops = []
    ops.append(np.sqrt(1 - p) * tensor(I, I))
    paulis = [X, Y, Z]
    for P in paulis:
        if target == 0:
            ops.append(np.sqrt(p / 3) * tensor(P, I))
        else:
            ops.append(np.sqrt(p / 3) * tensor(I, P))
    return sum(K * rho * K.dag() for K in ops)


def apply_depolarizing_both(rho, p):
    rho = depolarize(rho, p, 0)
    rho = depolarize(rho, p, 1)
    return rho

# General QFI calculator for mixed states
def qfi(rho, A):
    eigvals, eigvecs = rho.eigenstates()
    qfi_val = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            lam_i, lam_j = eigvals[i], eigvals[j]
            if lam_i + lam_j > 0:
                delta = (lam_i - lam_j)**2 / (lam_i + lam_j)
                element = complex((eigvecs[i].dag() * A * eigvecs[j]))
                qfi_val += delta * abs(element)**2
    return 4 * qfi_val

# Run simulation
ps = np.linspace(0, 1, 100)
qfi_data = {label: [] for label in generators}

for p in ps:
    if p == 0:
        print(f": QFI = ")

    for label, G in generators.items():
        # Apply unitary encoding
        U = (-1j * np.pi / 4 * G).expm()
        encoded = U * rho0 * U.dag()

        # Apply depolarizing noise
        noisy = apply_depolarizing_both(encoded, p)

        # Compute QFI
        qfi_val = qfi(noisy, G)
        qfi_data[label].append(qfi_val)

# Plot
for label, values in qfi_data.items():
    plt.plot(ps, values, label=label)

plt.xlabel("Depolarizing Strength p")
plt.ylabel("Quantum Fisher Information")
plt.title("QFI under Depolarizing Noise for Various Generators (Bell State)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
