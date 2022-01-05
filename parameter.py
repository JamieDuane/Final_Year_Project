import math
import numpy as np
from scipy.optimize import Bounds, minimize

class PARAMETER(object):
    def __init__(self, Xt, q):
        self.T = len(Xt[0, :])-1
        self.Xt = Xt
        self.d = len(Xt)
        self.q = q
        self.C = self.C()
        self.M = self.C * int(math.log(self.T, 10))
        self.s_r_hat = self.S_r_hat()
        self.w, self.yita = self.YITA()
        self.rho, self.beta = self.RHO_BETA()
        self.s_star = self.S_STAR()
        self.r, self.r_hat = self.R_HAT()
        self.s_hat = self.S_HAT()
        self.N = self.N()
        print(f'C   :{self.C}\nyita:{self.yita}\ns_hat:{self.s_hat}\ns_star:{self.s_star}\nR   :{self.r}\nR_hat:{self.r_hat}\nN   :{self.N}')

    def RHO_BETA(self):
        '''
        :param C: some constant defined somewhere
        :param T: period
        :param d: dimension of time series data
        :param q: compression dimension
        :return: rho and beta for ADMM initialization
        '''
        rho = self.C*(math.log(self.T, 10)**1.5)*(math.log(self.d, 10)/self.T)**0.5
        beta = rho*self.d/(self.q**0.5)
        return rho, beta

    def S_r_hat(self):
        s_r_hat = []
        for t in range(-self.M, self.M + 1):
            flag = 1
            if t < 0:
                flag = 0
                t = -t
            NtT = int((self.T - t) / (t + 1))
            #print(M, NtT, t)
            R_hat = np.zeros((self.d, self.d), dtype='complex_')
            for k in range(NtT + 1):
                if flag == 1:
                    R_hat += np.dot(self.Xt[:, (k + 1) * t + k].reshape((self.d, 1)), self.Xt[:, k * t + k].reshape((1, self.d)))
                else:
                    R_hat += np.dot(self.Xt[:, (k + 1) * t + k].reshape((self.d, 1)), self.Xt[:, k * t + k].reshape((1, self.d))).T.conj()
            s_r_hat.append((t, R_hat / (NtT + 1)))
        return s_r_hat

    def S(self, w):
        l = [complex(math.cos(2 * math.pi * w * t), -math.sin(2 * math.pi * w * t)) * e for t, e in self.s_r_hat]
        res = np.sum(l, axis=0)
        return res

    def YITA(self):
        bounds = Bounds([0], [1])
        res = minimize(self.rosen, np.array([1]), bounds=bounds)
        w = res.x
        return w, -self.rosen(w)

    def rosen(self, w):
        S = self.S(w[0])
        e, v = np.linalg.eig(S)
        idx = e.argsort()[::-1]
        evq = e[idx[self.q-1]]
        evq_1 = e[idx[self.q]]
        return -(3*evq_1+evq)/(evq_1+3*evq)

    def S_HAT(self):
        s_hat = self.C * max(4*self.q/(self.yita**(-0.5)-1)**(-2), 1)*self.s_star
        return s_hat

    def R_HAT(self):
        A = min(((2.*self.yita)**0.5)/4, (self.q*self.yita*(1-self.yita**0.5)/2)**0.5)
        r = ((self.q*(self.d**2)*(math.log(self.T, 10)**3)*math.log(self.d, 10)/self.T)**0.5) * (A-self.C*(math.log(self.T, 10)**1.5)*(math.log(self.d, 10)/self.T)**0.5)**(-1)
        r_hat = (4.+0.j)*(math.log(1./self.yita, 10)**(-1))*math.log(self.C*((self.yita/8)**0.5)*(math.log(self.T, 10))**1.5*(self.s_star*math.log(self.d, 10)/self.T)**0.5, 10)
        return int((r.real**2+r.imag**2)**0.5)+1, int((r_hat.real**2+r_hat.imag**2)**0.5)+1

    def N(self):
        print(math.log(self.T, 10), math.log(self.d, 10))
        n = self.C*(math.log(self.T, 10)**(-1.5))*(self.s_star*math.log(self.d, 10)/self.T)**(-0.5)
        return int(n)+1

    def C(self):
        return 90

    def S_STAR(self):
        S = self.S(0)
        res = 0
        for i in range(self.d):
            k = 0
            for j in range(len(S[i, :])):
                k += S[i, j]**2
            res += 1 if k != 0 else 0
        return res