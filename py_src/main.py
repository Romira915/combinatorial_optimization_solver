import math
import os
import string
from lib2to3.pgen2 import token

import dimod
import numpy as np
from dotenv import load_dotenv
from dwave.system import DWaveSampler, EmbeddingComposite
from numpy import hamming
from openjij import SQASampler
from pyqubo import Array, Constraint, LogEncInteger, solve_qubo

load_dotenv()

TOKEN = os.getenv("TOKEN")
endpoint = "https://cloud.dwavesys.com/sapi/"

W = 750
# c = {0: 5, 1: 7,  2: 2, 3: 1, 4: 4, 5: 3}
# w = {0: 8, 1: 10, 2: 6, 3: 4, 4: 5, 5: 3}
cost = [135, 139, 149, 150, 156, 163, 173,
        184, 192, 201, 210, 214, 221, 229, 240]
weight = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
opt = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
c = {}
w = {}
N = len(cost)
for i in range(N):
    c[i] = cost[i]
    w[i] = weight[i]

print(c)
print(w)


x = Array.create('x', shape=(N), vartype='BINARY')
y = LogEncInteger("y", (0, W))


key1 = max(c, key=lambda k: c[k])
B = 1
A = B * c[key1] * 2

HA = Constraint(
    A * (W - sum(w[a] * x[a] for a in range(N)) - y)**2, label='HA'
)

HB = - B * sum(c[a] * x[a] for a in range(N))


print("[Inputs]")
print()
print("W (ナップサックの容量) : "+str(W)+"kg")
print("N (宝物の数): "+str(N))
print()
print("weight list")
print(w)
print()
print("cost list")
print(c)
print()
print("A : "+str(A))
print("B : "+str(B))

H = HA + HB
Q = H
model = Q.compile()
q, offset = model.to_qubo()

# sampler = SQASampler()
dw_sampler = DWaveSampler(solver='Advantage_system4.1',
                          token=TOKEN, endpoint=endpoint)
print(dw_sampler.nodelist)
sampler = EmbeddingComposite(dw_sampler)
sampleset = sampler.sample_qubo(q, num_reads=2)
decoded_sample = model.decode_sample(sampleset.first.sample, vartype="BINARY")

print(sampleset.data_vectors)

print()
print("[Results]")
print(sampleset.record)
print()
print("decoded_sample.sample:")
print(decoded_sample.sample)
print()

weight = 0
cost = 0
d_hamming = 0

for k in range(N):
    d_hamming += abs(decoded_sample.array('x', k) - opt[k])
    if decoded_sample.array('x', k) != 0:
        print("宝物" + str(k))
        weight += w[k]
        cost += c[k]

d_hamming /= N

sol_y = sum(2**k * v for k, v in [(elem, decoded_sample.array('y', elem))
            for elem in range(math.ceil(math.log2(W)))])

print()
print("スラック変数Y = {}".format(sol_y))
print()
print("broken")
print(decoded_sample.constraints(only_broken=True))
print("合計の重さ : "+str(weight))
print("合計の価格 : "+str(cost))
print("ハミング距離 :" + str(d_hamming))

x = sampleset.record[-1][0]
x = np.append(x, [3 for _ in range(5627-len(x))])
print(x)
t = (0, 5, 15, 20)
s = (1.0, 0.4, 0.4, 1.0)

schedule = list(zip(t, s))

dw_sampler = DWaveSampler(solver='Advantage_system4.1',
                          token=TOKEN, endpoint=endpoint)
sampler = EmbeddingComposite(dw_sampler)
sampleset = sampler.sample_qubo(
    q, num_reads=2, anneal_schedule=schedule, initial_state=x)
print(sampleset.record)

weight = 0
cost = 0
d_hamming = 0

for k in range(N):
    d_hamming += abs(decoded_sample.array('x', k) - opt[k])
    if decoded_sample.array('x', k) != 0:
        print("宝物" + str(k))
        weight += w[k]
        cost += c[k]

d_hamming /= N

sol_y = sum(2**k * v for k, v in [(elem, decoded_sample.array('y', elem))
            for elem in range(math.ceil(math.log2(W)))])

print()
print("スラック変数Y = {}".format(sol_y))
print()
print("broken")
print(decoded_sample.constraints(only_broken=True))
print("合計の重さ : "+str(weight))
print("合計の価格 : "+str(cost))
print("ハミング距離 :" + str(d_hamming))
