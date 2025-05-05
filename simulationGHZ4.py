import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Pauli matrices
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state
ghz = (tensor(basis(2,0), basis(2,0), basis(2,0))
       + tensor(basis(2,1), basis(2,1), basis(2,1))).unit()
rho0 = ket2dm(ghz)

# Collective Z generator: G = (Z₁ + Z₂ + Z₃) / 2
G = (tensor(Z, I, I) + tensor(I, Z, I) + tensor(I, I, Z)) / 2

# Encoding unitary: U = exp(-i π/4 G)
U = (-1j * np.pi/4 * G).expm()

# Amplitude-damping Kraus operators on one qubit
def amplitude_damping_operators(gamma):
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])
    return [E0, E1]

# Apply amplitude damping independently to all 3 qubits
def apply_amplitude_damping_all(rho, gamma):
    kraus = amplitude_damping_operators(gamma)
    result = rho
    for i in range(3):
        new_rho = 0
        for K in kraus:
            if i == 0:
                K_full = tensor(K, I, I)
            elif i == 1:
                K_full = tensor(I, K, I)
            else:
                K_full = tensor(I, I, K)
            new_rho += K_full * result * K_full.dag()
        result = new_rho
    return result

# QFI with respect to G, summing only over i<j
# F = 4 * sum_{i<j} ((λ_i - λ_j)^2 / (λ_i + λ_j)) |<i|G|j>|^2
def qfi(rho, G):
    eigvals, eigvecs = rho.eigenstates()
    F = 0
    contrib = []
    N = len(eigvals)
    for i in range(N):
        for j in range(i):
            lam_i, lam_j = eigvals[i], eigvals[j]
            if lam_i + lam_j > 0:
                delta = (lam_i - lam_j)**2 / (lam_i + lam_j)
                element = complex((eigvecs[i].dag() * G * eigvecs[j]))
                term = 4 * delta * abs(element)**2
                F += term
                contrib.append(((i, j), term))
    return F, sorted(contrib, key=lambda x: x[1], reverse=True)

# Sample a few gamma values
gamma_values = [0.0, 0.5, 0.8, 1.0]

qfi_values = []
top_contribs = []

for γ in gamma_values:
    # encode, then apply amplitude damping
    encoded = U * rho0 * U.dag()
    noisy = apply_amplitude_damping_all(encoded, γ)

    # compute QFI and top contributions
    F, contrib = qfi(noisy, G)
    qfi_values.append(F)
    top_contribs.append(contrib[:5])

# Print results
for γ, F, contrib in zip(gamma_values, qfi_values, top_contribs):
    print(f"γ = {γ:.1f} → QFI = {F:.6f}")
    print("  Top contributions:")
    for (i, j), val in contrib:
        print(f"    (i={i}, j={j}): {val:.3e}")
    print()

# Plot QFI vs gamma
plt.figure(figsize=(6,4))
plt.plot(gamma_values, qfi_values, 'o-', label='QFI(γ)')
plt.axhline(9, linestyle='--', label='Ideal QFI (γ=0) = 9')
plt.xlabel('Amplitude Damping γ')
plt.ylabel('Quantum Fisher Information')
plt.title('QFI of GHZ under Amplitude Damping')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
