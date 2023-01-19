# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:01:46 2022

@author: Miguel
"""


import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
  "text.usetex": True,
    "font.family": "sans-serif"
})

def least_squares(x, y):
    
    if not isinstance(x, list):
        x = np.array(list(x))
    if not isinstance(y, list):
        y = np.array(list(y))
    x = np.array(x)
    y = np.array(y)
    
    assert len(x) == len(y), f"least_squares Error: len(y)={len(y)} /={len(x)}"
    assert len(x) > 1,       f"least_squares Error: {len(x)} = 1, 0"

    n = len(x)
    sx  = sum(x)
    sy  = sum(y)
    sxy = sum(x*y)
    sx2 = sum(x*x)
    
    A = (n*sxy - (sx*sy)) / (n*sx2 - (sx**2))
    B = (sy - (A*sx)) / n
    
    aux = y - (A*x) - B
    errA = (n/(n-2)) * (sum(aux*aux)) / (n*sx2 - (sx**2))
    errA = errA**0.5
    errB = errA * ((sx2 / n)**.5)

    
    return A, B, errA, errB

#%% Importing fields 

file_ = 'fields_matrix.txt'

subfold_ = '4He_SPSD'
subfold_ = '16O_SPSDPF'#'/1step'
# subfold_ = '16O_SPSDPF_alp1t1000'
#subfold_ = '8He_SPSD'

HAS_REA = False

gammaLR, gammaLR_DD = [],[]
DeltaLR, DeltaLR_DD, DeltaRL, DeltaRL_DD = [],[],[],[]
ReaField_DD = []
hspLR, hspLR_dd, hspLR_rea = [],[],[]

with open(f'fields_matrices/{subfold_}/{file_}', 'r') as f:
    data = f.readlines()
    SP_dim = int((len(data) - 1)**.5 )
    
    gammaLR     = np.zeros([SP_dim, SP_dim]) 
    gammaLR_DD  = np.zeros([SP_dim, SP_dim])
    DeltaLR     = np.zeros([SP_dim, SP_dim])
    DeltaLR_DD  = np.zeros([SP_dim, SP_dim])
    DeltaRL     = np.zeros([SP_dim, SP_dim])
    DeltaRL_DD  = np.zeros([SP_dim, SP_dim])
    ReaField_DD = np.zeros([SP_dim, SP_dim])
    hspLR       = np.zeros([SP_dim, SP_dim])
    hspLR_dd    = np.zeros([SP_dim, SP_dim])
    hspLR_rea   = np.zeros([SP_dim, SP_dim])
    
    for i, line in enumerate(data):
        if i==0: 
            continue
        
        line = line.split()
        if len(line) < 12:
            line.append(.0)
        else:
            HAS_REA = True
        line[0] = int(line[0])
        line[1] = int(line[1])
        # for i in range(2, len(line)):
        #     line[i] = float(line[i])
        
        i, j = line[:2]
        i -= 1
        j -= 1
        gammaLR[i,j]	 = float(line[2])
        gammaLR_DD[i,j]	 = float(line[3])
        DeltaLR[i,j]	 = float(line[4])
        DeltaLR_DD[i,j]	 = float(line[5])
        DeltaRL[i,j]	 = float(line[6])
        DeltaRL_DD[i,j]	 = float(line[7])
        ReaField_DD[i,j] = float(line[8])
        hspLR[i,j]       = float(line[9])
        hspLR_dd[i,j]	 = float(line[10])
        hspLR_rea[i,j]	 = float(line[11])
    

fig = plt.figure()
plt.imshow(gammaLR_DD)
plt.colorbar()
plt.show()

rea_sum = 0.0
for i in range(SP_dim):
    for j in range(i, SP_dim):
        rea_sum += ReaField_DD[i,j]
rea_sum = np.trace(ReaField_DD)
print("Rea_sum=",rea_sum)


fig = plt.figure()
X_gammaDD, Y_gammaRea = [], []
gamma_ij_labels = {}

index_ = 0
for i in range(SP_dim):
    for j in range(i, SP_dim):
        # if abs(gammaLR_DD[i,j]) < 1.0e-15:
        #     continueHF
        X_gammaDD.append(gammaLR_DD[i,j])
        Y_gammaRea.append(ReaField_DD[i,j])
        gamma_ij_labels[index_] = (i, j)
        
        index_ += 1

A, B, errA, errB = least_squares(Y_gammaRea, 
                                 X_gammaDD)
x_Min, x_Max = 0.9*min(Y_gammaRea), 1.1*max(Y_gammaRea)
dx = (x_Max - x_Min) / 20
x  = np.array([x_Min + (dx*i) for i in range(21)])
y  = (A*x) + B
yM = ((A+errA)*x) + B
ym = ((A-errB)*x) + B

plt.plot(x, y, 'c-', label=r'least squares $\Gamma=A·\partial\Gamma + B$', markersize = 1.0)
plt.plot(x, ym, 'c--', markersize = 1.0)
plt.plot(x, yM, 'c--', markersize = 1.0)
plt.scatter(Y_gammaRea, X_gammaDD, s=3.0, c='k')

plt.ylabel(r'$\Gamma^{DD}_{ij}$')
plt.xlabel(r'$\partial\Gamma^{DD}_{ij}$')
plt.legend()
plt.title(r'Comparison of DD Fields $^{16}$O on SPSDPF shell space.'
          + f"\nA={A:6.5f} $\pm$ {errA:6.5f}, B={B:7.6f} $\pm$ {errB:7.6f}")
plt.show()
print(f"A={A:6.5f} +- {errA:6.5f}, B={B:7.6f} +- {errB:7.6f}")
        

print(np.trace(ReaField_DD), np.trace(gammaLR_DD))

