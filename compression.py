import json, math
import numpy as np
import quadpy
from CSOAP import CSOAP

def get_b_c(U, t, N):
    shape = U.shape
    U = U.reshape(shape[1] * shape[2], shape[0])
    def f_b_c(w):
        down_k, up_k = np.floor(w * N).astype(int), np.ceil(w * N).astype(int)
        Uw1 = (U[:, down_k] / 2 - U[:, up_k] / 2)
        Uw2 = np.cos(math.pi * N * w - np.floor(w * N))
        Uw3 = U[:, down_k] / 2 + U[:, up_k]
        Uw = np.multiply(Uw2, Uw1).reshape(shape[1] * shape[2], len(w)) + Uw3
        # complex(math.cos(2 * math.pi * w * t), -math.sin(2 * math.pi * w * t))
        angle = 2 * math.pi * w * t
        complex_part = np.vectorize(complex)(np.cos(angle), -np.sin(angle))
        Uw = np.multiply(complex_part, Uw)
        return Uw
    bc, e = quadpy.quad(f_b_c, 0, 1, epsabs=1e-1, epsrel=1e-1, limit=1000)
    return bc

def get_info_loss(dic, res):
    def f_b_c(w):
        k = []
        j = 1
        for i in w:
            S = res.S(i)
            l = [complex(math.cos(2 * math.pi * i * t), -math.sin(2 * math.pi * i * t))*e for t, e in dic]
            O = np.sum(l, axis=0)
            B = O.reshape(res.d, res.q).T.conj()
            C = O.reshape(res.d, res.q)
            I = np.diag([1. + 0.j] * res.d)
            Q = I - np.dot(C, B)
            QT = Q.T.conj()
            mtrx = np.linalg.multi_dot([Q, S, QT])
            k.append(complex(np.trace(mtrx)))
            print(f'Round {j}/{len(w)}: {k[-1]}')
            j += 1
        return k
    loss, _ = quadpy.quad(f_b_c, 0, 1)
    print(loss)
    return loss

def compress(Xt):
    res = CSOAP(Xt=Xt, q=5)
    U = res.csoap()
    Xc = np.zeros(Xt.shape, dtype='complex_')
    dic = {}
    dic2 = []
    for t in range(res.T+1):
        sum_c = np.zeros((res.d,), dtype='complex_')
        for u in range(-res.N+t, res.N+t+1):
            sum_b = np.zeros((res.q,), dtype='complex_')
            for s in range(0, min(res.T+1, res.N+u+1)):
                if u-s in dic:
                    o = dic[u-s]
                else:
                    o = get_b_c(U, u-s, res.N)
                    dic[u-s] = o
                b = o.reshape(res.d, res.q).T.conj()
                Xb = np.dot(b, Xt[:, s])
                sum_b = sum_b+Xb
            if t-u in dic:
                o = dic[t-u]
            else:
                o = get_b_c(U, t-u, res.N)
                dic[t-u] = o
            c = o.reshape(res.d, res.q)
            sum_c = np.add(sum_c, np.dot(c, sum_b))
        #print(sum_c.shape)
            if t not in dic:
                dic[t] = get_b_c(U, t, res.N)
            dic2.append((t, dic[t]))
        Xc[:, t] += sum_c
        print(f'Compression round {t} done')
    print(Xc)
    print(Xc.shape)
    return dic2, res, Xc

def compress_multiple_file():
    l = [i for i in range(500, 5100, 100)]
    dic = {i:{'T': l.copy(), 'info_loss':[]} for i in [100, 150, 200]}
    for d in [100, 150, 200]:
        for t in range(500, 5100, 100):
            with open(f'data/d{d}T{t}.npy', 'rb') as f:
                Xt = np.load(f)
            bc_dic, res, Xc = compress(Xt)
            info_loss = get_info_loss(bc_dic, res)
            dic[d]['info_loss'].append(info_loss)
    return dic

if __name__ == '__main__':
    dic = compress_multiple_file()

    '''
    with open('data/d200T5000.npy', 'rb') as f:
        Xt = np.load(f)
        # Xt = Xt.T
    dic, res, Xc = compress(Xt)
    with open('data/compress.npy', 'wb') as f:
        np.save(f, Xc)
    '''
