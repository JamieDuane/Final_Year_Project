import numpy as np

class ADMMRelaxation(object):
    def __init__(self, cov, rho, beta, q):
        self.cov = cov
        self.d = len(cov)
        self.phi = np.zeros((self.d,self.d))
        self.pi = np.zeros((self.d, self.d))
        self.shita = np.zeros((self.d,self.d))
        self.rho = rho
        self.beta = beta
        self.q = q

    def get_ei_val_vec(self, mtrx):
        e, v = np.linalg.eig(mtrx)
        idx = e.argsort()[::-1]
        return e[idx], v[idx]

    def piforward(self): # Projection
        mtrx=self.phi+self.shita/self.beta+self.cov/self.beta
        ei_val, Q = self.get_ei_val_vec(mtrx)
        #print(mtrx)
        ei_val = np.real(ei_val)
        #print(ei_val.shape, self.d, mtrx.shape)
        # minimize 1/2 xTPx + px
        # subject to Gx <= h; Ax = b; lb <= x <= ub
        from qpsolvers import solve_qp
        Pc = np.diag([2.]*self.d)
        pl = np.dot(np.array([-2.]), ei_val.reshape((1,self.d))).reshape((self.d,))
        G = np.array([0.]*self.d)
        h = np.array([0.])
        A = np.array([1.]*self.d)
        b = np.array([self.q])
        lb = np.array([0.]*self.d)
        ub = np.array([1.]*self.d)
        #print(self.q, self.d)
        #print(Pc.shape, pl.shape, G.shape, h.shape, A.shape, b.shape, lb.shape, ub.shape)
        sol = solve_qp(Pc, pl, G, h, A, b, lb, ub)
        #print(sol)
        # Retrive Q NewEig Q-1:
        self.pi = np.dot(np.dot(Q, np.diag(sol)), np.linalg.inv(Q))

    #phit = np.array([[1,3],[2,6]])
    #shitat = np.array([[2,7],[3,8]])
    #cov = np.array([[2,3],[8,20]])
    #beta = 2
    #print(piforward(phit, shitat, cov, 2))

    def phiforward(self): # Soft Thresholding
        for i in range(self.d):
            for j in range(self.d):
                check = self.pi[i, j]-self.shita[i, j]/self.beta
                if abs(check) <= self.rho/self.beta:
                    self.phi[i, j] = 0
                else:
                    sign = 1 if check > 0 else -1 if check < 0 else 0
                    self.phi[i, j] = sign * (abs(check)-self.rho/self.beta)

    def shitaforward(self):
        self.shita = np.subtract(self.shita, self.rho * np.subtract(self.pi, self.phi))

    def initialize(self, R):
        res = np.zeros((self.d, self.d))
        for i in range(R):
            self.piforward()
            self.phiforward()
            self.shitaforward()
            res = np.add(res, self.pi)
            print(f'Initialization round {i+1} done')
        pi = res/R
        ei_val, ei_vec = self.get_ei_val_vec(pi)
        return ei_vec[:, :self.q]