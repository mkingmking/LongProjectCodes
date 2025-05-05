import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define Pauli Z operator and identity
iZ = qeye(2)
Z = sigmaz()

# Initial pure state |+> = (|0> + |1>)/sqrt(2)
psi0 = (basis(2, 0) + basis(2, 1)).unit()
rho0 = ket2dm(psi0)

# Parameterized unitary U(theta) = exp(-i theta Z/2)
def U(theta):
    return (-1j * theta * Z / 2).expm()

# Single-qubit amplitude-damping channel via Kraus operators
def amplitude_damp(rho, gamma):
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])
    return E0 * rho * E0.dag() + E1 * rho * E1.dag()

# QFI for mixed state with respect to generator A
# Uses eigen-decomposition, sum_{i<j}, and standard prefactor 4

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
    return 4 * q

# Scan amplitude damping gamma and compute QFI at theta=pi/4
gammas = np.linspace(0, 1, 100)
qfis = []
for gamma in gammas:
    # Prepare encoded state
    rho_theta = U(np.pi/4) * rho0 * U(-np.pi/4)
    # Apply amplitude-damping
    rho_noisy = amplitude_damp(rho_theta, gamma)
    # Compute QFI w.r.t. generator Z/2
    qfis.append(qfi(rho_noisy, Z/2))

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(gammas, qfis, label='QFI vs γ')
plt.axhline(1, linestyle='--', label='Ideal QFI (γ=0)')
plt.xlabel('Amplitude Damping γ')
plt.ylabel('Quantum Fisher Information')
plt.title('QFI of |+> state under amplitude damping')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
