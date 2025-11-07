import numpy as np

### ODE as a function and initial ocndition
y0 = 1
def f(y,x):
    return (x+y) / x
### Independent Variable Discretization
x_start = 1
x_end = 10
h = 0.01
x = np.arange(x_start, x_end + h, h)

### Euler Method Implementation and RK4 Method Implementation
def Eulermethod(f,y0,x):
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(0,len(x)-1):
        y[i+1] = y[i] + f(y[i],x[i]) * (x[i+1]-x[i])
    return y

Y_Euler = Eulermethod(f,y0,x)

def RKmethod(f,y0,x):
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(0,len(x)-1):
        h = x[i+1]-x[i]
        F1 = h*f(y[i],x[i])
        F2 = h*f((y[i]+F1/2),(x[i]+h/2))
        F3 = h*f((y[i]+F2/2),(x[i]+h/2))
        F4 = h*f((y[i]+F3),(x[i]+h))
        y[i+1] = y[i] + 1/6*(F1 + 2*F2 + 2*F3 + F4)
    return y

Y_RK = RKmethod(f,y0,x)

print(f"Final Euler value at x=10: {Y_Euler[-1]}")
print(f"Final RK4 value at x=10:   {Y_RK[-1]}")


