import numpy as np
from qutip import *

# Pauli matrices
I, X, Y, Z = qeye(2), sigmax(), sigmay(), sigmaz()

# GHZ state
ghz = (tensor(basis(2,0), basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1), basis(2,1))).unit()
rho0 = ket2dm(ghz)

# Generators
G_zzz = tensor(Z, Z, Z)
G_xxx = tensor(X, X, X)
Jz = tensor(Z, I, I) + tensor(I, Z, I) + tensor(I, I, Z)

# Depolarizing noise
def depolarize(rho, p, target):
    ops = [np.sqrt(1 - p) * tensor(I, I, I)]
    for P in [X, Y, Z]:
        if target == 0:
            ops.append(np.sqrt(p/3) * tensor(P, I, I))
        elif target == 1:
            ops.append(np.sqrt(p/3) * tensor(I, P, I))
        elif target == 2:
            ops.append(np.sqrt(p/3) * tensor(I, I, P))
    return sum(K * rho * K.dag() for K in ops)

def apply_depolarizing_all(rho, p):
    for i in range(3):
        rho = depolarize(rho, p, i)
    return rho

# QFI computation
def qfi(rho, G):
    eigvals, eigvecs = rho.eigenstates()
    val = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            if eigvals[i] + eigvals[j] > 0:
                delta = (eigvals[i] - eigvals[j])**2 / (eigvals[i] + eigvals[j])
                element = complex((eigvecs[i].dag() * G * eigvecs[j]))
                val += delta * abs(element)**2
    return 4 * val

# Sample p values
for p in [0.0, 0.5, 0.8, 1.0]:
    rho = apply_depolarizing_all(rho0, p)
    print(f"\n--- p = {p} ---")
    print("Eigenvalues:", np.round(rho.eigenenergies(), 6))
    print("QFI (Z⊗Z⊗Z):", qfi(rho, G_zzz))
    print("QFI (X⊗X⊗X):", qfi(rho, G_xxx))
    print("QFI (Jz):", qfi(rho, Jz))

# Compare with completely mixed state
mixed = (1/8) * tensor(I, I, I)

print("\n--- Completely Mixed State ---")
print("Eigenvalues:", np.round(mixed.eigenenergies(), 6))
print("QFI (Z⊗Z⊗Z):", qfi(mixed, G_zzz))
print("QFI (X⊗X⊗X):", qfi(mixed, G_xxx))
print("QFI (Jz):", qfi(mixed, Jz))
