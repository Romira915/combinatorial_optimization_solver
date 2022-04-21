from lib2to3.pgen2 import token
import math
import string
import dimod
from pyqubo import solve_qubo
from pyqubo import Constraint
from pyqubo import Array, LogEncInteger
from openjij import SQASampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("TOKEN")
endpoint = "https://cloud.dwavesys.com/sapi/"

W = 995
# c = {0: 5, 1: 7,  2: 2, 3: 1, 4: 4, 5: 3}
# w = {0: 8, 1: 10, 2: 6, 3: 4, 4: 5, 5: 3}
cost = [
    94, 506, 416, 992, 649, 237, 457, 815, 446, 422, 791, 359, 667, 598, 7, 544, 334, 766, 994,
    893, 633, 131, 428, 700, 617, 874, 720, 419, 794, 196, 997, 116, 908, 539, 707, 569, 537,
    931, 726, 487, 772, 513, 81, 943, 58, 303, 764, 536, 724, 789,
]
weight = [485, 326, 248, 421, 322, 795, 43, 845, 955, 252, 9, 901, 122, 94, 738, 574, 715, 882, 367,
          984, 299, 433, 682, 72, 874, 138, 856, 145, 995, 529, 199, 277, 97, 719, 242, 107, 122, 70,
          98, 600, 645, 267, 972, 895, 213, 748, 487, 923, 29, 674, ]
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
A = B * c[key1] * 5

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
dw_sampler = DWaveSampler(solver='Advantage_system4.1', token=TOKEN)
sampler = EmbeddingComposite(dw_sampler)
sampleset = sampler.sample_qubo(q, num_reads=50)
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
