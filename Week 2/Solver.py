import numpy as np
import numpy.linalg as al
from scipy.linalg import lu, solve
import copy

def JacobiSolver(A, b, x0, e, NO):
    # --- Input Validation ---
    if A.shape[0] != A.shape[1]:
        print("Error: Matrix A must be square.")
        return None
    if b.shape[0] != A.shape[0]:
        print("Error: Vector b size does not match matrix A.")
        return None
    if x0.shape[0] != A.shape[0]:
        print("Error: Initial guess x0 size does not match matrix A.")
        return None

    # --- Initialization ---
    n = len(b)
    x = copy.deepcopy(x0) 
    k = 1                  
    new_line = "\n"

    # --- Main Iteration Loop ---
    while (k <= NO):
        for i in range(n): 
            sum_a = 0
            for j in range(n):
                if j != i:
                    sum_a = sum_a + A[i][j] * x0[j] 
            
            # This line is now safe because b[i] is a scalar
            x[i] = (-sum_a + b[i]) / A[i][i]

        # Calculate relative error
        diff_inf = al.norm(x - x0, np.inf) / (al.norm(x, np.inf) + 1e-10) # Added 1e-10 to prevent divide by zero
        
        print(f"Jacobi Iteration {k}: {x.T}")

        # Check for convergence
        if (diff_inf < e):
            print(f"\nJacobi convergence achieved after {k} iterations.")
            print(f"The solution of the system is: {new_line}{x}")
            return x  
        
        k = k + 1
        x0 = copy.deepcopy(x) 

    # --- Failure Case ---
    if (k == NO + 1):
        print(f"\nFailure: Jacobi did not converge after {NO} iterations.")
        return None 

def GaussSeidelSolver(A, b, x0, e, NO): 

    # --- Input Validation ---
    if A.shape[0] != A.shape[1]:
        print("Error: Matrix A must be square.")
        return None
    if b.shape[0] != A.shape[0]:
        print("Error: Vector b size does not match matrix A.")
        return None
    if x0.shape[0] != A.shape[0]:
        print("Error: Initial guess x0 size does not match matrix A.")
        return None
    
    n = len(b)
    
    # Checking for Diagonal dominance
    for i in range(n):
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag < row_sum:
            print("Not diagonally dominant, Gauss-Seidel may not converge.")
        if A[i][i] == 0:
            print(f"Error: Zero on diagonal")
            return None

    x = copy.deepcopy(x0) 
    new_line = "\n"
            
    # --- Main Iteration Loop ---
    for iteration in range(NO):
        x_old = copy.deepcopy(x) # Store old x for comparison
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])    # CREATING NEW VALVUES
            sum2 = np.dot(A[i, i+1:], x_old[i+1:]) # Use *old* values
            
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        print(f"Gauss-Seidel Iteration {iteration + 1}: {x.T}")

        diff_inf = al.norm(x - x_old, np.inf) / (al.norm(x, np.inf) + 1e-10)
        
        if (diff_inf < e):
            print(f"\nGauss-Seidel convergence achieved after {iteration + 1} iterations.")
            print(f"The solution of the system is: {new_line}{x}")
            return x
    
    print(f"\nFailure: Gauss-Seidel did not converge after {NO} iterations.")
    return None

def LUSolver(A, b):
    # Input validation
    if A.shape[0] != A.shape[1]:
        print("Matrix A must be square.")
        return
    if b.shape[0] != A.shape[0]:
        print("Error: Vector b size does not match matrix A.")
        return None

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
