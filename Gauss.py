# Gauss Elimination Method
# Checking for square matrix and the sizing of b
# create the augmented matrix and then form the upper triangular matrix
# then perform back substitution to get the solution of the system

import numpy as np

def gauss_elimination(A, b):
    if A.shape[0] != A.shape[1]:
        print("Matrix A must be square.")
        return

    if b.shape[1] > 1 or b.shape[0] != A.shape[0]:
        print("Matrix b is incorrectly sized")
        return

    # Initialization of variables
    n = len(b)
    i = 0
    j = i -1
    m = n - 1
    x = np.zeros(n)
    new_line = "\n"

    # Creating the Augmented Matrix
    augmented_matrix = np.concatenate((A, b), axis=1, dtype=float)
    print(f"The Augmented Matrix is: {new_line}{augmented_matrix}")
    print("Upper Triangular Matrix:")

    # Gauss Elimination
    while i < n:
        if augmented_matrix[i][i] == 0.0:  # Avoid division by zero
            print("Divide by zero")
            return  

        for j in range(i + 1, n):
            scaling_factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] = augmented_matrix[j] - (scaling_factor * augmented_matrix[i])
            print(augmented_matrix)
        
        i += 1

    # Back Substitution
    x[m] = augmented_matrix[m][n] / augmented_matrix[m][m]

    for k in range(n-2, -1, -1):
        x[k] = augmented_matrix[k][n]

        for j in range(k+1, n):
            x[k] = x[k] - augmented_matrix[k][j] * x[j]
        x[k] = x[k] / augmented_matrix[k][k]


    print (f"The solution of the system is: {new_line}{x}")

# Input matrices here
variable_matrix = np.array([[0, 2, 4, 1], [-1, 0, 2, 5], [2, 3, 0, 5], [5, 1, 2 , 0]])
constant_matrix = np.array([[1], [-6], [8], [-4]])
gauss_elimination(variable_matrix, constant_matrix)