import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Pauli operators
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state |GHZ> = (|000> + |111>)/sqrt(2)
ghz = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) 
       + tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()
rho0 = ket2dm(ghz)

# Collective three-qubit generator G_coll = (Z1 + Z2 + Z3)/2
G_coll = (tensor(Z, I, I) + tensor(I, Z, I) + tensor(I, I, Z)) / 2

# Depolarizing noise on one qubit in a 3-qubit system
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

# Corrected QFI computation for mixed states (sum only over i<j)
def qfi(rho, A):
    eigvals, eigvecs = rho.eigenstates()
    q = 0
    N = len(eigvals)
    for i in range(N):
        for j in range(i):
            lam_i, lam_j = eigvals[i], eigvals[j]
            if lam_i + lam_j > 0:
                delta = (lam_i - lam_j)**2 / (lam_i + lam_j)
                # inner product yields a scalar complex
                element = eigvecs[i].dag() * A * eigvecs[j]
                q += delta * abs(element)**2
    # Use the standard prefactor so that for pure states F = 4 Var(A)
    return 4 * q

# Simulation over depolarizing strength p
ps = np.linspace(0, 1, 100)
qfi_values = []
for p in ps:
    # Encode with unitary evolution under G_coll (rotation by pi/4)
    U = (-1j * np.pi / 4 * G_coll).expm()
    encoded = U * rho0 * U.dag()
    noisy = apply_depolarizing_all(encoded, p)
    qfi_values.append(qfi(noisy, G_coll))

# Plot QFI vs depolarizing strength
plt.figure(figsize=(8, 6))
plt.plot(ps, qfi_values, label='QFI under G_coll')
plt.axhline(9, linestyle='--', label='Ideal QFI (p=0)')
plt.xlabel('Depolarizing strength p')
plt.ylabel('Quantum Fisher Information')
plt.title('QFI of GHZ state under depolarizing noise')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show() 
ps = np.linspace(0, 1, 100)
qfi_values = []
for p in ps:
    # Encode with unitary evolution under G_coll (rotation by pi/4)
    U = (-1j * np.pi / 4 * G_coll).expm()
    encoded = U * rho0 * U.dag()
    noisy = apply_depolarizing_all(encoded, p)
    qfi_values.append(qfi(noisy, G_coll))

# Plot QFI vs depolarizing strength
plt.figure(figsize=(8, 6))
plt.plot(ps, qfi_values, label='QFI under G_coll')
plt.axhline(9, linestyle='--', label='Ideal QFI (p=0)')
plt.xlabel('Depolarizing strength p')
plt.ylabel('Quantum Fisher Information')
plt.title('QFI of GHZ state under depolarizing noise')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
