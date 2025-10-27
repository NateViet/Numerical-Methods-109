import Solver
import numpy as np


# Question 2
# ==============================================================================================
variable_matrix = np.array([
    [0, 0, 1, -1],
    [0, 1, -4, 1],
    [1, -4, 1, 0],
    [-1, 1, 0, 0]
])
constant_matrix = np.array([
    7,
    -18,
    12,
    3
])

Solver.LUSolver(variable_matrix, constant_matrix)
print("\n" + "="*30 + "\n") # Separator

# Question 4
# ==============================================================================================
A = np.array([
    [5, 0.1, 0.3, -1],
    [0.5, 3, -0.4, 0.1],
    [0.1, -0.4, 5, 0],
    [-0.2, 0.1, 0, 5]
], dtype=float) 

b = np.array([1, 
    2, 
    3, 
    4
], dtype=float)

x0 = np.array([0, 0, 0, 0], dtype=float)  # Initial guess

NO = 1000                                 # Max iterations
e = 1.0e-3                                # Error tolerance

Solver.JacobiSolver(A,b,x0,e,NO)

print("\n" + "="*30 + "\n") # Separator

Solver.GaussSeidelSolver(A,b,x0,e,NO)

