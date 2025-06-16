import numpy as np
from qiskit.quantum_info import DensityMatrix, random_density_matrix
from qiskit.quantum_info.operators import Operator
import pandas as pd

# ── PARAMETERS ────────────────────────────────────────────────────────────────
n_qubits    = 2
dim         = 2**n_qubits
num_states  = 2000            # total examples in dataset
shots       = 1024            # number of measurement shots per basis setting

# Define single-qubit basis change unitaries for X, Y, Z
H = np.array([[1,  1],
              [1, -1]]) / np.sqrt(2)
Sdg = np.array([[1, 0],
                [0, -1j]])
bases = {
    'X': H,
    'Y': Sdg @ H,   # S†H maps Z→Y
    'Z': np.eye(2)
}

# Build the 9 joint-Pauli settings (and their unitaries)
settings = []
for b1 in ['X','Y','Z']:
    for b2 in ['X','Y','Z']:
        U = np.kron(bases[b1], bases[b2])
        settings.append((f"{b1}{b2}", Operator(U)))

# ── ALLOCATE DATA MATRICES ────────────────────────────────────────────────────
# X: num_states × (9 settings × 4 outcomes)
X = np.zeros((num_states, len(settings)*4))
# Y: store raw density matrices, then flatten
Y = np.zeros((num_states, dim, dim), dtype=complex)

# ── GENERATE DATA ─────────────────────────────────────────────────────────────
for i in range(num_states):
    # 1) sample a random 2-qubit density matrix (mixed)
    dm = random_density_matrix(dim)
    Y[i] = dm.data

    feats = []
    # 2) for each joint-Pauli setting, rotate & sample counts
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

# ── PREPARE TARGET VECTOR ─────────────────────────────────────────────────────
# flatten real+imag parts of each rho into a 1D target
Y_real = Y.real.reshape(num_states, -1)
Y_imag = Y.imag.reshape(num_states, -1)
y = np.hstack([Y_real, Y_imag])   # shape (num_states, 2*dim*dim)



M, F = X.shape
_,  T = y.shape
combined = np.hstack([X, y])   # shape (M, F+T)

# --- make column names ---
x_cols = [f"X_{i}"     for i in range(F)]
y_cols = [f"Y_real_{i}" for i in range(T//2)] + \
         [f"Y_imag_{i}" for i in range(T//2)]
cols   = x_cols + y_cols

# --- build DataFrame and write ---
df = pd.DataFrame(combined, columns=cols)
df.to_csv("qst_dataset.csv", index=False)