import numpy as np

k1 = 1
k2 = 2 
k3 = 3

C0 = np.array([1.0, 0.0, 0.0])  # Initial concentrations: [A], [B], [C] at t=0

h = 0.1
t_start = 0
t_end = 10
t = np.arange(t_start, t_end + h, h)

def species(t, C):
    CA = C[0]
    CB = C[1]
    CC = C[2]

    dCA_dt = -k1 * CA
    dCB_dt = k1 * CA - k2 * CB + k3 * CC
    dCC_dt = k2 * CB - k3 * CC

    return np.array([dCA_dt, dCB_dt, dCC_dt])

def EulerPC(f, C0, t):
    n = len(t)
    C_history = np.zeros((n, len(C0)))
    C_history[0] = C0
    
    for i in range(n-1):
        h = t[i+1] - t[i]
        C_i = C_history[i]
        #Predictor
        k1 = f(t[i], C_i)
        C_pred = C_i + h * k1
        
        # Corrector step (k2)
        k2 = f(t[i+1], C_pred)
        
        # Average the slopes
        C_history[i+1] = C_i + (h / 2.0) * (k1 + k2)
        
    return C_history

def RKMethod(f, C0, t):
    n = len(t)
    C_history = np.zeros((n, len(C0)))
    C_history[0] = C0
    
    for i in range(n - 1):
        h = t[i+1] - t[i]
        C_i = C_history[i]
        
        # Calculate the four 'k' vectors (slopes)
        k1 = f(t[i], C_i)
        k2 = f(t[i] + h/2.0, C_i + (h/2.0) * k1)
        k3 = f(t[i] + h/2.0, C_i + (h/2.0) * k2)
        k4 = f(t[i] + h, C_i + h * k3)
        
        C_history[i+1] = C_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return C_history

C_heun = EulerPC(species, C0, t)
C_rk4 = RKMethod(species, C0, t)

C_heun_final = C_heun[-1]
C_rk4_final = C_rk4[-1]

#PRINT
print("\n(b) Euler Predictor-Corrector:")
print(f"    C_A(10) ≈ {C_heun_final[0]:.8f}")
print(f"    C_B(10) ≈ {C_heun_final[1]:.8f}")
print(f"    C_C(10) ≈ {C_heun_final[2]:.8f}")

print("\n(c) 4th-Order Runge-Kutta (RK4):")
print(f"    C_A(10) ≈ {C_rk4_final[0]:.8f}")
print(f"    C_B(10) ≈ {C_rk4_final[1]:.8f}")
print(f"    C_C(10) ≈ {C_rk4_final[2]:.8f}")
