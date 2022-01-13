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

#print(quadpy.quad(integrand, 0, 1)[0])

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

def FFT():
    x = np.array([1, 2, 3, 4, 5, 7])
    y1 = np.fft.fft(x, n=6)
    def intergrand(w):
        xw = np.zeros((len(w),), dtype='complex_')
        for i in range(len(x)):
            angle = 2 * math.pi * w * i
            complex_part = np.vectorize(complex)(np.cos(angle), -np.sin(angle))
            xw = np.add(np.multiply(complex_part, x[i]), xw)
        return xw
    y2 = quadpy.quad(intergrand, 0, 1)[0]

    print(sum(y1))
    print(y2)

#FFT()

def try_cov():
    x = np.random.normal(loc=0, size=5)
    y = np.random.normal(loc=0, size=5)
    cov = np.cov(x, y)
    print(x)
    print(y)
    print(cov)
    print(np.cov(y, x))
try_cov()