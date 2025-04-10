import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define Pauli operators
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state
ghz = (tensor(basis(2,0), basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1), basis(2,1))).unit()
rho0 = ket2dm(ghz)

# Generators
generators = {
    "X⊗X⊗X": tensor(X, X, X),
    "Y⊗Y⊗Y": tensor(Y, Y, Y),
    "Z⊗Z⊗Z": tensor(Z, Z, Z),
    "X⊗Y⊗Z": tensor(X, Y, Z),
    "Y⊗X⊗Z": tensor(Y, X, Z),
}

# Depolarizing noise
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

# Amplitude damping noise
def amplitude_damping_operators(gamma):
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])
    return [E0, E1]

def apply_amplitude_damping_all(rho, gamma):
    kraus_ops = amplitude_damping_operators(gamma)
    result = rho
    for i in range(3):  # for each qubit
        new_rho = 0
        for K in kraus_ops:
            if i == 0:
                K_full = tensor(K, I, I)
            elif i == 1:
                K_full = tensor(I, K, I)
            else:
                K_full = tensor(I, I, K)
            new_rho += K_full * result * K_full.dag()
        result = new_rho
    return result

# QFI function
def qfi(rho, G):
    eigvals, eigvecs = rho.eigenstates()
    val = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            lam_i, lam_j = eigvals[i], eigvals[j]
            if lam_i + lam_j > 0:
                delta = (lam_i - lam_j)**2 / (lam_i + lam_j)
                element = complex((eigvecs[i].dag() * G * eigvecs[j]))
                val += delta * abs(element)**2
    return 4 * val

# Simulate for both channels
ps = np.linspace(0, 1, 100)
results_dep = {}
results_amp = {}

for label, G in generators.items():
    dep_vals = []
    amp_vals = []
    for p in ps:
        U = (-1j * np.pi / 4 * G).expm()
        encoded = U * rho0 * U.dag()
        dep_rho = apply_depolarizing_all(encoded, p)
        amp_rho = apply_amplitude_damping_all(encoded, p)
        dep_vals.append(qfi(dep_rho, G))
        amp_vals.append(qfi(amp_rho, G))
    results_dep[label] = dep_vals
    results_amp[label] = amp_vals

# Plotting
fig, axs = plt.subplots(len(generators), 1, figsize=(4, 7))

for i, label in enumerate(generators):
    axs[i].plot(ps, results_dep[label], label="Depolarizing", color='tab:blue')
    axs[i].plot(ps, results_amp[label], label="Amplitude Damping", color='tab:red', linestyle='--')
    axs[i].set_title(f"QFI for Generator {label}")
    axs[i].set_xlabel("Noise Strength (p or gamma)")
    axs[i].set_ylabel("QFI")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()
