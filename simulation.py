import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define initial state |+> = (|0> + |1>)/sqrt(2)
psi0 = (basis(2, 0) + basis(2, 1)).unit()

# Define parameterized unitary evolution (for QFI)
def U(theta):
    return (-1j * theta * sigmaz() / 2).expm()

# Function to apply amplitude damping
def amplitude_damped_state(theta, gamma):
    psi = U(theta) * psi0
    rho = ket2dm(psi)
    c_ops = [np.sqrt(gamma) * destroy(2)]
    return mesolve(H=0 * qeye(2), rho0=rho, c_ops=c_ops, tlist=[0, 1]).states[-1]


# Function to compute QFI (for a pure state)
def qfi_z(rho):
    # Compute the symmetric logarithmic derivative (SLD)
    eigvals, eigvecs = rho.eigenstates()
    qfi = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            if eigvals[i] + eigvals[j] > 0:
                dij = (eigvals[i] - eigvals[j])**2 / (eigvals[i] + eigvals[j])
                matrix_element = complex((eigvecs[i].dag() * sigmaz() * eigvecs[j]))


                qfi += dij * np.abs(matrix_element)**2
    return 4 * qfi

# Scan gamma values
gammas = np.linspace(0, 1, 100)
qfis = []

for gamma in gammas:
    rho = amplitude_damped_state(theta=np.pi/4, gamma=gamma)
    qfis.append(qfi_z(rho))

# Plot
plt.plot(gammas, qfis)
plt.xlabel("Amplitude Damping γ")
plt.ylabel("Quantum Fisher Information")
plt.title("QFI vs Amplitude Damping")
plt.grid(True)
plt.show()
