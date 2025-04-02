here are the simulations done with qutip library for the quantum metrology on a remote network project.



What is QFI?
Quantum Fisher Information (QFI) measures how sensitively a quantum state rho(theta) depends on a parameter theta that is encoded via a unitary evolution:

rho(theta) = exp(-i * theta * G) * rho_0 * exp(i * theta * G)

Where:

G is the generator (a Hermitian operator),

theta is the parameter to estimate,

rho_0 is the original state.

F_Q = 4 * Var(G) = 4 * (⟨G^2⟩ - ⟨G⟩^2)

For mixed states, the expression is more involved but fundamentally captures the same idea: how much the state changes under small changes in theta.
