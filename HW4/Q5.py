import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, fsolve

# Part (a): Analytical Solution
def analytical_solution(x):
    return 1 / x

def shooting_method():
    # Define the ODE system
    # Let y[0] = y and y[1] = y'
    # y'' = 2y^3 => y[1]' = 2 * y[0]^3
    def ode_system(x, y):
        return [y[1], 2 * y[0]**3]
    
    # Boundary conditions
    x_start, x_end = 1, 2
    y_start, y_end_target = 1, 0.5

    # Objective function: Find the initial slope
    def objective(alpha):
        # We need to solve the IVP with a guess slope, named alpha
        sol = solve_ivp(ode_system, [x_start, x_end], [y_start, alpha], rtol=1e-6)
        y_final_calculated = sol.y[0, -1]
        return y_final_calculated - y_end_target
    
    # Find the root (the correct initial slope)
    # We guess the slope is negative because the y goes from 1 --> 0.5
    root_result = root_scalar(objective, bracket=[-2, 0], method='brentq')
    correct_slope = root_result.root

    print(f"Shooting Method: Found initial slope y'(1) = {correct_slope:.5f}")

    # Solve one last time wit hthe correct slope to get the plots
    t_eval = np.linspace(x_start, x_end, 100)
    final_sol = solve_ivp(ode_system, [x_start, x_end], [y_start, correct_slope], t_eval=t_eval, rtol=1e-6)

    return final_sol.t, final_sol.y[0]

# Part (c) : Finite Difference Method

def finite_difference_method():
    # Domain and Grid

    x_start, x_end = 1, 2
    N_interior = 5 # given

    # total = Interior +2 bound
    N_total = N_interior + 2
    x_nodes = np.linspace(x_start, x_end, N_total)
    h = x_nodes[1] - x_nodes[0]

    # Function to solve F(Y_interior) = 0
    def residuals(y_interior):

        y = np.zeros(N_total)
        y[0] = 1.0
        y[-1] = 0.5
        y[1:-1] = y_interior

        # We need to calculate the residuals for the discretized ODE

        res = []
        for i in range(1, N_total - 1):
            finite_diff = (y[i+1] - 2 * y[i] + y[i-1]) / (h**2)
            term = 2 * y[i]**3
            res.append(finite_diff - term)

        return np.array(res)
    
    # Initial guess for interior points (linear interpolation)

    y_guess = np.linspace(1.0, 0.5, N_total)[1:-1]

    y_interior_solution = fsolve(residuals, y_guess)

    # Reconstruct full solution array for plotting
    y_sol_full = np.zeros(N_total)
    y_sol_full[0] = 1.0
    y_sol_full[-1] = 0.5
    y_sol_full[1:-1] = y_interior_solution

    return x_nodes, y_sol_full

# Plotting Results
# 1. Analytical Solution
x_analytical = np.linspace(1, 2, 100)
y_analytical = analytical_solution(x_analytical)

# 2. Shooting Method
x_shooting, y_shooting = shooting_method()

# 3. Finite Difference Method
x_fd, y_fd = finite_difference_method()

plt.figure(figsize=(10, 6)) 
#plot analytical
plt.plot(x_analytical, y_analytical, 'k--', label='Analytical Solution', linewidth=2)
#plot shooting
plt.plot(x_shooting, y_shooting, 'r--', label='Shooting Method', markersize=5)
#plot finite difference
plt.plot(x_fd, y_fd, 'bo-', label='Finite Difference Method', markersize=5)

plt.title("Comparison of Analytical and Numerical Solutions ($y'' - 2y^3 = 0$)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

