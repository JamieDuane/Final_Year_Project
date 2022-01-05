from initualizing_process import ADMMRelaxation
import numpy as np
import json
from parameter import PARAMETER

class CSOAP(PARAMETER):
    def __init__(self, Xt, q):
        super().__init__(Xt, q)

    def truncnate(self, V_dcmp):
        def get_vi_list_2(v):
            res = 0
            for e in v:
                res += (e.real**2 + e.imag**2)**0.5
            return res
        U = np.zeros(V_dcmp.shape, dtype='complex_')
        vi_list = np.zeros((self.d,))
        for i in range(self.d):
            v = V_dcmp[i, :]
            vi_list[i] = get_vi_list_2(v)
        idx = vi_list.argsort()[::-1]
        for j in range(min(self.s_hat, self.d)):
            index = idx[j]
            U[index,:] = V_dcmp[index,:]
        return U

    def thin_QR(self, V):
        v, R = np.linalg.qr(V)
        return v, R

    def soap(self, S, U):
        for t in range(self.r_hat):
            V = np.dot(S, U)
            V_dcmp, R1 = self.thin_QR(V)
            U_hat = self.truncnate(V_dcmp)
            U, R2 = self.thin_QR(U_hat)
        return U

    def csoap(self):
        pi = ADMMRelaxation(cov=self.S(0), rho=self.rho, beta=self.beta, q=self.q)
        Uinit = pi.initialize(self.r)
        print('Initialization Process Done')
        U_hat0 = self.truncnate(Uinit)
        Uinit, v = self.thin_QR(U_hat0)
        #print(Uinit.shape)
        U_pre = self.soap(self.S(0), Uinit)
        res = [U_pre]
        for i in range(1, self.N+1):
            U_next = self.soap(self.S(i/self.N), U_pre)
            res.append(U_next)
            U_pre = U_next
            print(f'CSOAP round {i} done')
        print(f'CSOAP Process Done')
        return np.array(res)

if __name__ == '__main__':
    with open('data/data.json', 'r') as f:
        data = json.load(f)
        Xt = np.array(data)
        #Xt = Xt.T
    print(Xt.shape)
    #(1000, 1000) (1000,) (1000,) (1,) (1000,) (1,) (1000,) (1000,)
    res = CSOAP(Xt=Xt, q=5)
    U = res.csoap()
    print(U.shape)