import random
import json
import numpy as np

def dataset_generator_v1():
    ts = [[10-6.8,8-6.8,3-6.8,7-6.8,6-6.8]*40, [-7,1,6,-8,4,4]*33+[-7,1], [1-2.5,2-2.5,3-2.5,4-2.5]*50, [1-1.5,2-1.5]*100, [3-2,2-2,1-2]*66+[3-2,2-2], [0]*200]
    start_list = random.sample(range(1, 20001), 400)
    result = []
    for i in range(400):
        check = i%6
        l = [start_list[i]*k for k in ts[check]]
        result.append(l)
    with open('data/data.json', 'w') as f:
        json.dump(result, f)

def dataset_generator_v2(T, d, q, s_star, f_name='data'):
    # Xt = AF[t] + e[t]
    # define A
    A = np.zeros((d, q))
    for i in range(s_star):
        A[i, :] = np.random.normal(0, 1, q)

    # define F[t]
    # F[t] = DF[t-1] + shita[t-1]
    shita = np.zeros((q, T+1))
    F = np.zeros((q, T+1))
    for i in range(T+1):
        if i == 0:
            F[:, i] = np.random.normal(loc=0, size=q)
        else:
            F[:, i] = 0.8 * F[:, i-1] + shita[:, i-1]
        shita[:, i] = np.random.normal(loc=0, scale=1, size=q)

    # define e[t]
    et = np.zeros((d, T+1))
    for i in range(d):
        et[i, :] = np.random.normal(0, 1, T+1)

    Xt = np.add(np.dot(A, F), et)
    #print(Xt.shape)

    with open(f'data/{f_name}.npy', 'wb') as f:
        np.save(f, Xt)

if __name__ == '__main__':
    for d in [100, 150, 200]:
        for t in range(500, 5100, 100):
            name = f'd{d}T{t}'
            dataset_generator_v2(T=t, d=d, q=10, s_star=100, f_name=name)