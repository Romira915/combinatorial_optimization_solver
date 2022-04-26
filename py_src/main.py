import math
import os
import string
from lib2to3.pgen2 import token

import dimod
from dotenv import load_dotenv
from dwave.system import DWaveSampler, EmbeddingComposite
from openjij import SQASampler
from pyqubo import Array, Constraint, LogEncInteger, solve_qubo

load_dotenv()

TOKEN = os.getenv("TOKEN")
endpoint = "https://cloud.dwavesys.com/sapi/"

W = 26
# c = {0: 5, 1: 7,  2: 2, 3: 1, 4: 4, 5: 3}
# w = {0: 8, 1: 10, 2: 6, 3: 4, 4: 5, 5: 3}
cost = [24, 13, 23, 15, 16]
weight = [12, 7, 11, 8, 9]
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
B = 10
A = B * c[key1] * 40

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

sampler = SQASampler()
# dw_sampler = DWaveSampler(solver='Advantage_system4.1', token=TOKEN)
# sampler = EmbeddingComposite(dw_sampler)
sampleset = sampler.sample_qubo(q, num_reads=10)
decoded_sample = model.decode_sample(sampleset.first.sample, vartype="BINARY")
print()
print("[Results]")
print(sampleset.record)
print()
print("decoded_sample.sample:")
print(decoded_sample.sample)
print()

weight = 0
cost = 0

for k in range(N):
    if decoded_sample.array('x', k) != 0:
        print("宝物" + str(k))
        weight += w[k]
        cost += c[k]


sol_y = sum(2**k * v for k, v in [(elem, decoded_sample.array('y', elem))
            for elem in range(math.ceil(math.log2(W)))])

print()
print("スラック変数Y = {}".format(sol_y))
print()
print("broken")
print(decoded_sample.constraints(only_broken=True))
print("合計の重さ : "+str(weight))
print("合計の価格 : "+str(cost))
