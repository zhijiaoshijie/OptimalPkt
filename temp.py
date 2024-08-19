import numpy as np

# Coefficients of the trigonometric polynomial
a = [1,1j,-1,-1j,1]
n= 5

# Construct the companion matrix
n = len(a) - 1
C = np.zeros((n, n))
C[1:, :-1] = np.eye(n-1)
C[:, -1] = -np.array(a[1:])

# Find the eigenvalues
eigenvalues = np.linalg.eigvals(C)

# Convert eigenvalues to angles
roots = np.angle(eigenvalues)
print(roots)