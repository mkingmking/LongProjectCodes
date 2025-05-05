import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Bell state |Φ+> = (|00> + |11>)/sqrt(2)
bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(bell)

# Pauli operators
iI = qeye(2)
X = sigmax()
Y = sigmay()
Z = sigmaz()

# Z-rotation only on first qubit: U(theta) = exp(-i theta Z/2 ⊗ I)
def U(theta):
    return tensor((-1j * theta * Z / 2).expm(), iI)

# Depolarizing noise on individual qubits
def depolarize_single_qubit(rho, p, target):
    ops = [np.sqrt(1 - p) * tensor(iI, iI)]
    for P in (X, Y, Z):
        if target == 0:
            ops.append(np.sqrt(p / 3) * tensor(P, iI))
        else:
            ops.append(np.sqrt(p / 3) * tensor(iI, P))
    return sum(K * rho * K.dag() for K in ops)

def depolarize_both(rho, p):
    for t in (0, 1):
        rho = depolarize_single_qubit(rho, p, t)
    return rho

# Phase-damping (dephasing) Kraus operators
def phase_damping_ops(gamma):
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, 0], [0, np.sqrt(gamma)]])
    return [E0, E1]

def apply_phase_damping(rho, gamma, target):
    ops = phase_damping_ops(gamma)
    new_rho = 0
    for K in ops:
        K_full = tensor(K, iI) if target == 0 else tensor(iI, K)
        new_rho += K_full * rho * K_full.dag()
    return new_rho

def apply_phase_damping_both(rho, gamma):
    for t in (0, 1):
        rho = apply_phase_damping(rho, gamma, t)
    return rho

# QFI w.r.t. A = Z ⊗ I: sum over i<j, prefactor 4 for pure-state limit
def qfi_z(rho):
    A = tensor(Z, iI)
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

# Simulation over noise strength
gammas = np.linspace(0, 1, 100)
qfis_dep = []
qfis_phase = []

for g in gammas:
    encoded = U(np.pi/4) * rho0 * U(-np.pi/4)
    qfis_dep.append(qfi_z(depolarize_both(encoded, g)))
    qfis_phase.append(qfi_z(apply_phase_damping_both(encoded, g)))

# Plot
plt.figure(figsize=(8, 6))
plt.plot(gammas, qfis_dep, label='Depolarizing')
plt.plot(gammas, qfis_phase, '--', label='Phase Damping')
plt.axhline(4, linestyle='--', label='Ideal QFI=4')
plt.xlabel('Noise strength (p or γ)')
plt.ylabel('Quantum Fisher Information')
plt.title('QFI of Bell state under Depol. & Phase-Damping')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
