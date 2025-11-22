## rewritten (d^2/dt^2) = x - dx/dt - x^2 + t 
# System of equations let x = x1 and x2 = dx/dt
# dx1/dt = x2
# dx2/dt = x1 - x2 - x1^2 + t

import numpy as np

def solve_ode_rk4():
    
    h = 0.1 # step size
    t_start = 0
    t_end = 10

    # Initial Conditions: x(0) = 1, x'(0) = 1
    x = 1.0
    v = 1.0 # v = dx/dt

    t_values = np.arange(t_start, t_end + h, h)

    def get_derivatives(t, current_x, current_v):
        dxdt = current_v #dx/dt = v
        dvdt = current_x - current_v - current_x**2 + t #dv/dt = x - v - x^2 + t , where v is dx/dt
        return np.array([dxdt, dvdt])

    # RK4 Integration Loop
    for i in range(len(t_values)-1):
        t = t_values[i]
        
        k1 = get_derivatives(t, x, v)

        k2 = get_derivatives(t + 0.5*h, x + 0.5*h*k1[0], v + 0.5*h*k1[1])
        
        k3 = get_derivatives(t + 0.5*h, x + 0.5*h*k2[0], v + 0.5*h*k2[1])
        
        # k4: Slopes at the end of the interval
        k4 = get_derivatives(t + h, x + h*k3[0], v + h*k3[1])

        slope_x = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
        slope_v = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0

        x = x + h * slope_x
        v = v + h * slope_v

    print(f"Solution x(10): {x:.6f}")

    return t_values, x

solve_ode_rk4()