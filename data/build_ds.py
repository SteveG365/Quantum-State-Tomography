import numpy as np
from qiskit.quantum_info import DensityMatrix, random_density_matrix
from qiskit.quantum_info.operators import Operator
import pandas as pd

#define params
n_qubits    = 2
dim         = 2**n_qubits
num_states  = 2000  
shots       = 1024  # shots per basis

# Define single-qubit basis change unitaries for X, Y, Z
H = np.array([[1,  1],
              [1, -1]]) / np.sqrt(2)
Sdg = np.array([[1, 0],
                [0, -1j]])

#store the basis operators in dictionary
bases = {
    'X': H,
    'Y': Sdg @ H,
    'Z': np.eye(n_qubits)
}

# Build the 9 joint-Pauli settings and Unitaries
settings = []
for b1 in ['X','Y','Z']:
    for b2 in ['X','Y','Z']:
        U = np.kron(bases[b1], bases[b2])
        settings.append((f"{b1}{b2}", Operator(U)))

# X: stores the counts which will become raw frequencies
X = np.zeros((num_states, len(settings)*4))
# Y: store raw density matrices, then flatten
Y = np.zeros((num_states, dim, dim), dtype=complex)

# Data Generation
for i in range(num_states):
    # sample a random 2-qubit density matrix (mixed)
    dm = random_density_matrix(dim)
    Y[i] = dm.data

    feats = []
    # for each joint-Pauli setting, rotate & sample counts
    for name, U in settings:
        # rotate into computational (Z⊗Z) basis
        rotated_dm = DensityMatrix(dm).evolve(U)
        # get ideal probs for 4 outcomes
        probs = rotated_dm.probabilities()
        # simulate shot noise
        counts = np.random.multinomial(shots, probs)
        # optionally store frequencies instead:
        freqs = counts / shots
        feats.extend(freqs)

    X[i] = feats

# flatten real+imag parts of each rho into a 1D target
Y_real = Y.real.reshape(num_states, -1)
Y_imag = Y.imag.reshape(num_states, -1)
y = np.hstack([Y_real, Y_imag])   # shape (num_states, 2*dim*dim)



M, F = X.shape
_,  T = y.shape
combined = np.hstack([X, y])

# Build and output the Dataframe into a csv file
x_cols = [f"X_{i}"     for i in range(F)]
y_cols = [f"Y_real_{i}" for i in range(T//2)] + \
         [f"Y_imag_{i}" for i in range(T//2)]
cols   = x_cols + y_cols

df = pd.DataFrame(combined, columns=cols)
df.to_csv("qst_dataset.csv", index=False)