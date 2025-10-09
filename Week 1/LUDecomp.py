# LU Decomposition Method
# Augmented Matrix to LU = A, where the scaling factors are in the lower triangular matrix
#Ax=b -> LUx=b -> Ly=b -> Ux=y
import numpy as np
from scipy.linalg import lu, solve

def lu_decomposition(A, b):
    # Input validation
    if A.shape[0] != A.shape[1]:
        print("Matrix A must be square.")
        return
    if b.shape[1] > 1 or b.shape[0] != A.shape[0]:
        print("Matrix b is incorrectly sized")
        return

    # Initialization
    n = len(b)
    new_line = "\n"

    # Performing LU Decomposition (A = PLU)
    P, L, U = lu(A)
    print(f"The Lower Triangular Matrix is: {new_line}{L}")
    print(f"The Upper Triangular Matrix is: {new_line}{U}")

    # 1. Solve for c in Pc = b  -->  c = P.T * b
    # this is a little differnt than the traditional methosd as the scipy.linalg.lu function
    # returns are matric in a second form A = PLU so we need to adjust.
    # This is because this method uses partial pivoting so to get the original matrices in the form we will transpose P
    c = np.dot(P.T, b)

    # 2. Solve for y in Ly = c
    y = solve(L, c)

    # 3. Solve for x in Ux = y
    x = solve(U, y)

    print(f"The solution of the system is: {new_line}{x}")

    # Verification Step
    # Calculate A*x
    b_calculated = np.dot(A, x)

    # Print for comparison
    print(f"Original b vector: {new_line}{b}")
    print(f"Calculated b vector (A * x): {new_line}{b_calculated}")

    # Check if the calculated b is close to the original b
    if np.allclose(b, b_calculated):
        print("\nVerification successful! The solution is correct ")
    else:
        print("\nVerification failed. The solution is incorrect ")

    return x

variable_matrix = np.array([
    [0, 2, 4, 1],
    [-1, 0, 2, 5],
    [2, 3, 0, -1],
    [5, 1, 2, 0]
])
constant_matrix = np.array([
    [1],
    [-6],
    [8],
    [-4]
])

lu_decomposition(variable_matrix, constant_matrix)

# In the example given, it is a singular matrix