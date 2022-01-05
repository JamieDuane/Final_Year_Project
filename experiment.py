import numpy as np
import quadpy, math

N = 2
U = np.array([[[1.]*4, [1.]*4, [2.]*4]]*3)
#print(U)
U = U.reshape(12, 3)
#print(U)
def integrand(x):
    print(len(x))
    k = []
    for i in x:
        k.append((2+3j)*i)
    return k

print(quadpy.quad(integrand, 0, 1)[0])

def inter(x):
    k = np.vectorize(complex)(np.cos(2 * math.pi * x), -np.sin(2 * math.pi * x))
    #print(k.shape)
    return k

#print(quadpy.quad(inter, 0, 1))

def solver(vars):
    return [
        sum(i**2 for i in vars),
        sum(2*i for i in vars)-10
    ]

from scipy.optimize import fsolve
#print(fsolve(solver, [0, 0]))