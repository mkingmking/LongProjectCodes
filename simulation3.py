import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Bell state |Φ+> = (|00> + |11>)/sqrt(2)
bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(bell)


# Define Pauli operators for each qubit
I = qeye(2)
X = sigmax()
Y = sigmay()
Z = sigmaz()

# Z-rotation only on first qubit
def U(theta):
    return tensor((-1j * theta * Z / 2).expm(), I)

# Phase damping channel (Kraus operators)
def phase_damping_operators(gamma):
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, 0], [0, np.sqrt(gamma)]])
    return [E0, E1]

# Apply dephasing to one target qubit in 2-qubit system
def apply_phase_damping(rho, gamma, target):
    E = phase_damping_operators(gamma)
    result = 0
    for K in E:
        if target == 0:
            K_full = tensor(K, I)
        else:
            K_full = tensor(I, K)
        result += K_full * rho * K_full.dag()
    return result

# Apply to both qubits
def apply_phase_damping_both(rho, gamma):
    rho = apply_phase_damping(rho, gamma, 0)
    rho = apply_phase_damping(rho, gamma, 1)
    return rho

def depolarize_single_qubit(rho, p, target):
    ops = []
    ops.append(np.sqrt(1 - p) * identity([2, 2]))

    for P in [X, Y, Z]:
        if target == 0:
            ops.append(np.sqrt(p / 3) * tensor(P, I))
        else:
            ops.append(np.sqrt(p / 3) * tensor(I, P))

    return sum(K * rho * K.dag() for K in ops)

# Full depolarizing channel (each qubit independently)
def depolarize_both(rho, p):
    rho = depolarize_single_qubit(rho, p, target=0)
    rho = depolarize_single_qubit(rho, p, target=1)
    return rho

# QFI w.r.t. A = Z ⊗ I
def qfi_z(rho):
    A = tensor(Z, I)
    eigvals, eigvecs = rho.eigenstates()
    qfi = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            if eigvals[i] + eigvals[j] > 0:
                delta = (eigvals[i] - eigvals[j])**2 / (eigvals[i] + eigvals[j])
                matrix_element = complex((eigvecs[i].dag() * A * eigvecs[j]))
                qfi += delta * abs(matrix_element)**2
    return 4 * qfi

# Simulation
gammas = np.linspace(0, 1, 100)
qfis = []

for gamma in gammas:
    state = U(np.pi / 4) * rho0 * U(-np.pi / 4)
    noisy = depolarize_both(state, gamma)
    qfis.append(qfi_z(noisy))

# Plot
plt.plot(gammas, qfis)
plt.xlabel("Phase Damping γ")
plt.ylabel("Quantum Fisher Information")
plt.title("QFI of Bell State under Independent depolarizing")
plt.grid(True)
plt.show()
