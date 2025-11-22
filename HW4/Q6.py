import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

def solve_diffusion():
    # Parameters
    L = 1.0  # Length of the rod
    n_interior = 10
    n_total = n_interior + 2
    h = L / (n_total - 1)

    # Boundary conditions
    y1_0, y1_L = 1.0, 0.0  
    y2_0, y2_L = 0.0, 1.0

    # Initialization
    # Unknowns vector U containing interior points, Size 2 * n_interior

    y1_init = np.linspace(y1_0, y1_L, n_total)[1:-1]
    y2_init = np.linspace(y2_0, y2_L, n_total)[1:-1]
    U = np.concatenate([y1_init, y2_init])

    # Newton-Raphson Iteration
    max_iter = 20
    tol = 1e-8

    for it in range(max_iter):
        # Extract y1 and y2 interior arrays from current U
        y1 = U[:n_interior]
        y2 = U[n_interior:]
        
        # Initialize Residual Vector (R) and Jacobian Matrix (J)
        R = np.zeros(2 * n_interior)
        J = np.zeros((2 * n_interior, 2 * n_interior))
        
        inv_h2 = 1.0 / h**2
        
        # --- Construct System ---
        for i in range(n_interior):
            # Indices for the Jacobian
            idx_y1 = i                # Row for Eq 1
            idx_y2 = i + n_interior   # Row for Eq 2
            
            # -- Equation 1: y1'' - y1^2 + y2 = 0 --
            
            # Finite Difference part: (y1_{i+1} - 2y1_i + y1_{i-1}) / h^2
            val_y1 = y1[i]
            val_y1_prev = y1[i-1] if i > 0 else y1_0
            val_y1_next = y1[i+1] if i < n_interior - 1 else y1_L
            
            val_y2 = y2[i] # Coupling term
            
            # Residual calculation
            fd_term = (val_y1_next - 2*val_y1 + val_y1_prev) * inv_h2
            R[idx_y1] = fd_term - val_y1**2 + val_y2
            
            # Jacobian derivatives for Eq 1
            J[idx_y1, idx_y1] = -2*inv_h2 - 2*val_y1  # dR1/dy1_i
            if i > 0:
                J[idx_y1, idx_y1-1] = inv_h2          # dR1/dy1_{i-1}
            if i < n_interior - 1:
                J[idx_y1, idx_y1+1] = inv_h2          # dR1/dy1_{i+1}
            J[idx_y1, idx_y2] = 1.0                   # dR1/dy2_i (Coupling)

            # -- Equation 2: y2'' + y1^2 - y2 = 0 --
            
            # Finite Difference part
            val_y2_prev = y2[i-1] if i > 0 else y2_0
            val_y2_next = y2[i+1] if i < n_interior - 1 else y2_L
            
            # Residual calculation
            fd_term_2 = (val_y2_next - 2*val_y2 + val_y2_prev) * inv_h2
            R[idx_y2] = fd_term_2 + val_y1**2 - val_y2
            
            # Jacobian derivatives for Eq 2
            J[idx_y2, idx_y2] = -2*inv_h2 - 1.0       # dR2/dy2_i
            if i > 0:
                J[idx_y2, idx_y2-1] = inv_h2          # dR2/dy2_{i-1}
            if i < n_interior - 1:
                J[idx_y2, idx_y2+1] = inv_h2          # dR2/dy2_{i+1}
            J[idx_y2, idx_y1] = 2*val_y1              # dR2/dy1_i (Coupling)

        # J * delta = -R  => delta = -J \ R
        delta = solve(J, -R)
        
        # Update solution
        U = U + delta
        
        # Check convergence
        res_norm = norm(R, np.inf)
        print(f"Iteration {it+1}: Residual Norm = {res_norm:.2e}")
        
        if res_norm < tol:
            print("\nConverged!")
            break
            
    # Reconstruct full arrays including boundaries for plotting
    y1_sol_interior = U[:n_interior]
    y2_sol_interior = U[n_interior:]
    
    y1_final = np.concatenate(([y1_0], y1_sol_interior, [y1_L]))
    y2_final = np.concatenate(([y2_0], y2_sol_interior, [y2_L]))
    

    return y1_final, y2_final

solve_diffusion()