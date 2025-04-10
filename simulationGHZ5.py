import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Pauli matrices
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state and XXX generator
ghz = (tensor(basis(2,0), basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1), basis(2,1))).unit()
rho0 = ket2dm(ghz)
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

# QFI function
def qfi(rho, G):
    eigvals, eigvecs = rho.eigenstates()
    val = 0
    contrib = []
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            lam_i, lam_j = eigvals[i], eigvals[j]
            if lam_i + lam_j > 0:
                delta = (lam_i - lam_j)**2 / (lam_i + lam_j)
                element = eigvecs[i].dag() * G * eigvecs[j]  
                term = 4 * delta * abs(element)**2
                val += term
                contrib.append(((i, j), term))
    return val, sorted(contrib, key=lambda x: x[1], reverse=True)

# Prepare data at selected gamma values
gamma_values = [0.0, 0.5, 0.8, 1.0]
states = []
reduced_bloch = []
qfi_contribs = []
matrices = []

U = (-1j * np.pi / 4 * G).expm()

for gamma in gamma_values:
    encoded = U * rho0 * U.dag()
    noisy = apply_amplitude_damping_all(encoded, gamma)
    states.append(noisy)

    # Reduced state of qubit 0
    rho_A = noisy.ptrace(0)
    bloch_x = expect(X, rho_A)
    bloch_y = expect(Y, rho_A)
    bloch_z = expect(Z, rho_A)
    reduced_bloch.append((bloch_x, bloch_y, bloch_z))

    # QFI contributions
    qfi_val, contrib = qfi(noisy, G)
    qfi_contribs.append(contrib[:5])  # top 5 contributions

    # Matrix form (rounded for readability)
    matrices.append(np.round(noisy.full(), 3))

print("\nReduced Bloch Vectors (qubit 0):")
for g, bloch in zip(gamma_values, reduced_bloch):
    print(f"γ = {g:.1f} → ⟨X⟩ = {bloch[0]:.3f}, ⟨Y⟩ = {bloch[1]:.3f}, ⟨Z⟩ = {bloch[2]:.3f}")

print("\nDensity Matrices at Key γ:")
for g, mat in zip(gamma_values, matrices):
    print(f"\nγ = {g:.1f}")
    print(mat)

print("\nTop QFI-Contributing Terms:")
for g, contrib in zip(gamma_values, qfi_contribs):
    print(f"\nγ = {g:.1f}")
    for (i, j), val in contrib:
        print(f"  (i={i}, j={j}) → contribution: {val:.3e}")



import matplotlib.pyplot as plt


# Extract Bloch vector components
bloch_xs = [x for (x, y, z) in reduced_bloch]
bloch_ys = [y for (x, y, z) in reduced_bloch]
bloch_zs = [z for (x, y, z) in reduced_bloch]

# 3D plot
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the Bloch sphere surface for reference
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(xs, ys, zs, color='lightgray', alpha=0.1, linewidth=0)

# Plot Bloch vector trajectory
ax.plot(bloch_xs, bloch_ys, bloch_zs, marker='o', color='blue', label='Qubit 0')
for i, g in enumerate(gamma_values):
    ax.text(bloch_xs[i], bloch_ys[i], bloch_zs[i], f"γ={g:.1f}", size=8)

ax.set_xlabel("⟨X⟩")
ax.set_ylabel("⟨Y⟩")
ax.set_zlabel("⟨Z⟩")
ax.set_title("Trajectory of Qubit 0 on the Bloch Sphere")
ax.legend()
plt.tight_layout()
plt.show()

