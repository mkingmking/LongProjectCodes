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

# Define Z rotation on qubit 0
def U(theta):
    return tensor((-1j * theta * Z / 2).expm(), I)

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

# QFI with respect to Z ⊗ I
def qfi_z(rho):
    eigvals, eigvecs = rho.eigenstates()
    A = tensor(Z, I)
    qfi = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            if eigvals[i] + eigvals[j] > 0:
                dij = (eigvals[i] - eigvals[j])**2 / (eigvals[i] + eigvals[j])
                matrix_element = complex((eigvecs[i].dag() * A * eigvecs[j]))
                qfi += dij * abs(matrix_element)**2
    return 4 * qfi

# Simulation
ps = np.linspace(0, 1, 100)
qfis = []

for p in ps:
    state = U(np.pi / 4) * rho0 * U(-np.pi / 4)
    noisy = depolarize_both(state, p)
    qfis.append(qfi_z(noisy))

# Plot
plt.plot(ps, qfis)
plt.xlabel("Depolarizing strength p")
plt.ylabel("Quantum Fisher Information")
plt.title("QFI of Bell State under Depolarizing Noise")
plt.grid(True)
plt.show()
