import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define Pauli operators
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state
ghz = (tensor(basis(2,0), basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1), basis(2,1))).unit()
rho0 = ket2dm(ghz)

# Generator: XXX
G = tensor(X, X, X)

# Amplitude damping channel
def amplitude_damping_operators(gamma):
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])
    return [E0, E1]

def apply_amplitude_damping_all(rho, gamma):
    kraus_ops = amplitude_damping_operators(gamma)
    result = rho
    for i in range(3):
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

# Purity
def purity(rho):
    return (rho * rho).tr().real

# QFI
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

# Projection onto generator eigenbasis
def projection_on_generator_eigenbasis(rho, G):
    eigvals, eigvecs = G.eigenstates()
    projections = []
    for vec in eigvecs:
        proj = (vec.dag() * rho * vec).real
        projections.append(proj)
    return eigvals, projections

# Simulation
gammas = np.linspace(0, 1, 100)
purities = []
qfis = []
eigen_proj_data = []

# Pre-rotation
U = (-1j * np.pi / 4 * G).expm()

for gamma in gammas:
    encoded = U * rho0 * U.dag()
    noisy = apply_amplitude_damping_all(encoded, gamma)
    purities.append(purity(noisy))
    qfis.append(qfi(noisy, G))
    if np.round(gamma, 2) in [0.0, 0.5, 0.8, 1.0]:
        eigvals, projections = projection_on_generator_eigenbasis(noisy, G)
        eigen_proj_data.append((gamma, eigvals, projections))

# Plot QFI and purity
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(gammas, qfis, label="QFI (XXX)", color="tab:blue")
plt.xlabel("Amplitude Damping γ")
plt.ylabel("Quantum Fisher Information")
plt.title("QFI vs γ")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(gammas, purities, label="Purity", color="tab:orange")
plt.xlabel("Amplitude Damping γ")
plt.ylabel("Tr(ρ²)")
plt.title("Purity vs γ")
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot eigenbasis projections at key gamma values
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, (gamma_val, eigvals, projections) in enumerate(eigen_proj_data):
    axs[idx].bar(range(len(projections)), projections)
    axs[idx].set_title(f"Projection on XXX eigenbasis at γ = {gamma_val}")
    axs[idx].set_xlabel("Eigenstate index")
    axs[idx].set_ylabel("Population")
    axs[idx].set_xticks(range(len(projections)))

plt.tight_layout()
plt.show()
