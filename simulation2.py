import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Bell state |Φ+> = (|00> + |11>)/sqrt(2)
bell = (tensor(basis(2, 0), basis(2, 0)) +
        tensor(basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(bell)

# Pauli operators
iI = qeye(2)
X = sigmax()
Y = sigmay()
Z = sigmaz()

# Z-rotation on qubit 0: U(theta) = exp(-i theta Z/2 ⊗ I)
def U(theta):
    return tensor((-1j * theta * Z / 2).expm(), iI)

# Depolarizing noise on one qubit
# Uses ops based on full two-qubit identity tensor
def depolarize_single_qubit(rho, p, target):
    ops = [np.sqrt(1 - p) * tensor(iI, iI)]
    for P in (X, Y, Z):
        if target == 0:
            ops.append(np.sqrt(p / 3) * tensor(P, iI))
        else:
            ops.append(np.sqrt(p / 3) * tensor(iI, P))
    return sum(K * rho * K.dag() for K in ops)

# Apply depolarizing to both qubits independently
def depolarize_both(rho, p):
    rho = depolarize_single_qubit(rho, p, target=0)
    rho = depolarize_single_qubit(rho, p, target=1)
    return rho

# QFI with respect to A = Z ⊗ I
# Use eigen-decomposition, sum over i<j, and prefactor 4 for pure-state limit

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

# Simulation over depolarizing strength p
ps = np.linspace(0, 1, 100)
qfis = []

for p in ps:
    encoded = U(np.pi/4) * rho0 * U(-np.pi/4)
    noisy = depolarize_both(encoded, p)
    qfis.append(qfi_z(noisy))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(ps, qfis, label='QFI under Z⊗I')
plt.axhline(4, linestyle='--', label='Ideal QFI (p=0)')
plt.xlabel('Depolarizing strength p')
plt.ylabel('Quantum Fisher Information')
plt.title('QFI of Bell state under depolarizing noise')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
