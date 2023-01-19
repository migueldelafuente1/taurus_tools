# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:51:58 2022

@author: Miguel
"""

from copy import deepcopy
import os
import os
import sys
from time import time
from time import time, sleep

from matplotlib.colors import LogNorm
from matplotlib.pyplot import imshow, show
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sympy import S
from sympy.physics.quantum.cg import CG

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import numpy as np


FOLDER = 'MatrixElementsEvolution/z10n14_SPSD/'
FILE_templ = 'uncoupled_DD{}.2b'

data = {}
absval_data = {}

files = os.listdir(FOLDER)
files_step = [f.split('_')[-1][:-3] for f in files]
files_step.remove('DD')

sort_vals = False
HOspDim = 80 if 'PF/' in FOLDER else 40

for step in files_step:
    
    with open(FOLDER + FILE_templ.format('_'+step), 'r') as f:
        mes = f.readlines()[1:]
        
        absval_data[step] = [None]*len(mes)
        for i in range(len(mes)):
            me = mes[i]
            a,b,c,d, me = me.split()
            a,b,c,d = int(a), int(b), int(c), int(d)
            me = float(me)
            mes[i] = (tuple([a,b,c,d]), me)
            absval_data[step][i] = (tuple([a,b,c,d]), abs(me))
            
        data[step] = dict(mes)        
        absval_data[step] = dict(absval_data[step])
        
        if sort_vals:
            data[step] = sorted(data[step].items(), key=lambda item: item[1])
            absval_data[step] = sorted(absval_data[step].items(), 
                                           key=lambda item: item[1])

if sort_vals:
    for step in files_step:
        print('top values at step :', step)
        print('       max      ::', data[step][ 0])
        print('       min      ::', data[step][-1])
        
        print('       max (abs)::', absval_data[step][ 0])
        print('       min (abs)::', absval_data[step][-1])
    
    # order by the state
    for step in files_step:
        data[step]        = dict(sorted(data[step],        key= lambda x: x[0]))
        absval_data[step] = dict(sorted(absval_data[step], key= lambda x: x[0]))

#%% View evolution of certain groups of m.e
state_bra = (1, 41)

y_states  = {}
x_steps   = [int(x) for x in files_step] 

plt.figure()
sel_mes = filter(lambda x: x[:2] == state_bra, data[files_step[0]].keys())

for ist, st in enumerate(sel_mes):
    y_states[st] = []
    
    for step in files_step:
        y_states[st].append(data[step][st])
        
    plt.plot(x_steps, y_states[st])
    plt.text(x_steps[-1] + 0.001*(ist / len(x_steps)), 
             y_states[st][-1]  + 0.001*(ist / len(x_steps)), str(st[2:]), 
             horizontalalignment='center')
    

plt.title('Evolution of DD m.e. for bra= {}'.format(state_bra))
plt.xlabel('steps')
plt.ylabel('< {} {} | vDD| * * > [MeV]'.format(*state_bra))
plt.show()

#%% Maximum differences between start and end

qn_set    = set([x[:2] for x in data[files_step[0]]])
#qn_set.add((set[x[2:] for x in data[files_step[0]]])
qn_set = list(qn_set)

hamil = np.zeros([qn_set.__len__(), qn_set.__len__()])

step_0 = files_step[0]
step_N = files_step[-1]

for i in range(len(qn_set)):
    st_bra = qn_set[i]
    for j in range(i, len(qn_set)):
        st_ket = qn_set[j]
        
        curr_st = tuple([*st_bra, *st_ket])
        if not curr_st in data[step_0]:
            continue
        hamil[i, j] = (data[step_N][curr_st] - data[step_0][curr_st])
        # if hamil[i,j] > 0.001:
        #     print("hi", i, j, hamil[i, j])

plt.figure()
plt.grid()
plt.title("Difference vDD[{}] to [{}]  MAX/MIN={:6.5f} {:6.5f}"
          .format(step_0, step_N, hamil.max(), hamil.min()))
plt.imshow(hamil, cmap='bwr')
plt.colorbar()
plt.show()

#%% Maximum difference during the process

pass
    
    
    
    



j_max = 9
A = np.zeros(((j_max+1)//2, (j_max+1)//2), dtype=int)

for jb in range(1, j_max+1,2):
    lb = (jb+1) // 2
    
    for ja in range(1, j_max+1, 2):
        la = (ja+1) // 2
    
        j_top = max(la, lb)
        
        
        # index_ = ((j_top-1)**2) + lb*(j_top - lb) +  (j_top - la) + 1
        # index_ = ((j_top-1)**2) + la*(1 - (j_top == la)) +  (j_top - lb)
        if la < lb:
            index_ = ((lb-1)**2) + la
            print(ja, jb,"  ", la, lb, " -> ",index_)
        else:
            index_ = ((la-1)**2) + (la-lb) + la
            print(ja, jb,"  ", la, lb, " -> (",index_,")")
        
        A[la-1,lb-1] = index_
    print()
        
imshow(A)
show()
#%% Import Matrix Elements Uncoupled DD.txt on iterations (considering Nme unfixed)


MAINFOLDER = 'FieldAndME_tests/uncoupled_DD_ODD_'

FOLDER = MAINFOLDER + 'sph'
# FOLDER = MAINFOLDER + 'axi'
# FOLDER = MAINFOLDER + 'tri'
FOLDER = MAINFOLDER + 'deform'

FOLDER = MAINFOLDER + 'smallSpace'
# FOLDER = MAINFOLDER + 'bigSpace'

FOLDER = 'FieldAndME_tests/ddMatrixElementsExportable16O_sph'
FOLDER = 'FieldAndME_tests/ddMatrixElementsExportable16O_deform'

# FOLDER = MAINFOLDER + 'triax_seed5'
# FOLDER = MAINFOLDER + 'oct'
# FOLDER = MAINFOLDER + 'axi_oct'
# FOLDER = MAINFOLDER + 'tri_oct'


FOLDER += '/'

shell_ofSP = {}
sp_basis = {}
sh_basis = {}

class SP:
    ORBITAL = {0:'s', 1:'p', 2:'d', 3:'f', 4:'g', 5:'h'}
    
    def __init__(self,i_sp, n,l,j,m, mt, i_tr, i_sh):
        
        self.n = n
        self.l = l
        self.j = j
        self.mj = m
        self.mt = mt
        self.i_tr = i_tr
        self.i_sp = i_sp
        self.i_sh = i_sh
    
    def __str__(self):
        part = 'p' if self.mt == -1 else 'n'
        return f"{self.n}{self.ORBITAL[self.l]}_{self.j}/2({self.mj})({part})"

class SH:
    ORBITAL = {0:'s', 1:'p', 2:'d', 3:'f', 4:'g', 5:'h'}
    
    def __init__(self, n,l,j, i_sh):
        
        self.n = n
        self.l = l
        self.j = j
        self.i_sh = i_sh
        
        self.ind_ant = str(10000*n + 100*l + j) if (n,l,j)!=(0,0,1) else '001'
    
    def __str__(self):
        return f"{self.n}{self.ORBITAL[self.l]}_{self.j}/2"
        
def getBasisSP(data, separator=', '):
    global sp_basis, shell_ofSP
    
    for index, line in enumerate(data.split('\n')[:-1]):
        if index == 0: continue
        line = line.strip()
        # print(line.split(separator))
        i_sp, i_sh, n,l,j,m, mt,i_tr = line.split(separator)
        
        i_sp, i_sh, i_tr = int(i_sp), int(i_sh), int(i_tr)
        n,l,j,m, mt      = int(n),int(l),int(j),int(m), int(mt)
        sp_basis[i_sp]   = SP(i_sp, n,l,j,m, mt, i_tr, i_sh)
        shell_ofSP[i_sp] = i_sh
        
        if i_sh not in sh_basis:
            sh_basis[i_sh] = SH(n,l,j, i_sh)



stepOfelems = []
elements    = {}
par_conserv_elems = {}
par_breakin_elems = {}
mjtot_breakin_elems = {}


for file_ in os.listdir(FOLDER):
    if not file_.startswith('uncoupled_DD'): continue
    
    print(file_)
    
    step, _ = file_.replace('uncoupled_DD_', '').split('.')
    step = int(step) if step.isnumeric() else 0
    
    stepOfelems.append(step)
    elements[step] = {}
    par_conserv_elems[step] = {}
    par_breakin_elems[step] = {}
    mjtot_breakin_elems[step] = {}
    
    with open(FOLDER+file_, 'r') as f:
        _, spindx, data = f.read().split('//')
        
        if not sp_basis:
            getBasisSP(spindx)
        
        for line in data.split('\n')[1:-1]:
            a,b,c,d,V = line.split()
            a,b,c,d = int(a),int(b),int(c),int(d)
            V = float(V)
            elements[step][(a,b,c,d)] = V
            
            sumL = sp_basis[a].l + sp_basis[b].l   + sp_basis[c].l + sp_basis[d].l
            sumM = sp_basis[a].mj + sp_basis[b].mj - sp_basis[c].mj - sp_basis[d].mj
            if sumL % 2 == 0 and sumM == 0:
                par_conserv_elems[step][(a,b,c,d)] = V
            else:
                if sumL % 2 == 1:
                    par_breakin_elems[step][(a,b,c,d)]   = V
                if sumM != 0 and sumL % 2 == 0:
                    mjtot_breakin_elems[step][(a,b,c,d)] = V
                    
        topsBreakP = list(par_breakin_elems[step].values())
        topsBreakM = list(mjtot_breakin_elems[step].values())
        topsConsr  = list(par_conserv_elems[step].values())
        print(f"  CONSERV   min={min(topsConsr):+12.8f}  min|{np.abs(topsConsr).min():5.4e}|  max={max(topsConsr):12.8f}  [{len(topsConsr)}]")
        if len(topsBreakP):
            print(f"  BREAK PAR min={min(topsBreakP):+9.6e} min|{np.abs(topsBreakP).min():5.4e}|  max={max(topsBreakP):9.6e}  [{len(topsBreakP)}]")
        else:
            print(f"  BREAK PAR min={' None':15}  max={' None':15}  total ={len(topsBreakP)}")
        if len(topsBreakM):
            print(f"  BREAK MJt min={min(topsBreakM):+9.6e} min|{np.abs(topsBreakM).min():5.4e}|  max={max(topsBreakM):9.6e}  [{len(topsBreakM)}]")
        else:
            print(f"  BREAK MJt min={' None':15}  max={' None':15}  total ={len(topsBreakM)}")

            


FieldExplicit = np.zeros((len(sp_basis), len(sp_basis)))
Field         = np.zeros((len(sp_basis), len(sp_basis)))
DiffFields    =  None

with open(FOLDER+'fields_matrix.txt', 'r') as f: #(p+mj)
    data = f.readlines()[1:]
    
    for line in data:
        vals = line.split()
        i = int(vals[0])
        j = int(vals[1])
        gammaDD = float(vals[3])
        deltaDD = float(vals[5])
        reaDD   = float(vals[8])
        
        Field[i-1,j-1] = gammaDD

with open(FOLDER+'fields_matrix_explicit.txt', 'r') as f:
    data = f.readlines()[1:]
    
    for line in data:
        vals = line.split()
        i = int(vals[0])
        j = int(vals[1])
        gammaDD = float(vals[3])
        deltaDD = float(vals[5])
        reaDD   = float(vals[8])
                
        FieldExplicit[i-1,j-1] = gammaDD
    
    DiffFields = Field - FieldExplicit
    
if os.path.exists(FOLDER+'fields_matrix_explicit_generated.txt'):
    FieldsGenerated = np.zeros((len(sp_basis), len(sp_basis)))
    
    with open(FOLDER+'fields_matrix_explicit_generated.txt', 'r') as f:
        data = f.readlines()[1:]
        
        for line in data:
            line = line.split(',')
            vals = line[0].split()
            a = int(vals[0])
            c = int(vals[1])
            vals = line[1].split()
            b = int(vals[0])
            d = int(vals[1])
            
            vals = line[2].split()
            gammaDD = float(vals[0])
            gammaDD_dir = float(vals[1])
            gammaDD_exc = float(vals[2])
            
            vals = line[3].split()
            vDD     = float(vals[0])
            vDD_dir = float(vals[1])
            vDD_exc = float(vals[2])
            
            # TODO: get the permutations and the phase (complete explicitely the fields)
            # sum_ = 0.0
            # if a != c:
            #     if b != d:
            #         sum_ += 
            #     else:
            # else:
            
        
            # FieldsGenerated[a-1,c-1] += gammaDD
            FieldsGenerated[a-1,c-1] += gammaDD_dir - gammaDD_exc
        
        
        N = len(sp_basis)
        for a in range(1,len(sp_basis)//2):
            for c in range(c+1, len(sp_basis)//2):
                FieldsGenerated[c,a] = FieldsGenerated[a,c]
                FieldsGenerated[c+ N//2, a+ N//2] = FieldsGenerated[a,c]
                
    DiffFields = FieldsGenerated - FieldExplicit
    # DiffFields = FieldsGenerated - Field
    
# for i in range(len(sp_basis)):
#     for j in range(i, len(sp_basis)):
#         if abs(DiffFields[i,j]) >  1.e-5:
#             print(sp_basis[i+1], sp_basis[j+1], "=", DiffFields[i,j])

plt.figure()
plt.title(FOLDER)
plt.imshow(DiffFields)
plt.colorbar()

plt.show()




fig, (ax1, ax2) = plt.subplots(1,2)
plt.title(FOLDER)
ax1.set_title("Gamma from vDD explicitly. ")
ax1.imshow(FieldExplicit)
ax2.set_title("Gamma from Fields. ")
im3 = ax2.imshow(Field)
# Create divider for existing axes instance
divider3 = make_axes_locatable(ax2)
# Append axes to the right of ax3, with 20% width of ax3
cax3 = divider3.append_axes("right", size="10%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(10), format="%.1f")
# Remove xticks from ax3
# ax2.xaxis.set_visible(False)
# Manually set ticklocations
# ax2.set_yticks([0.0, 2.5, 3.14, 4.0, 5.2, 7.0])

plt.show()



#%% SCRIPT TO OBTAIN THE J reduced ME AND TENSOR COMPONENTS 


hamilJM = {-2:{}, 0:{}, 2:{}}

step = 0
Nstep = 1000
t1 = time()

for k, vdd in elements[0].items():
    step += 1
    if step % Nstep == 0:
        t1 = time() - t1
        print(step, "/",len(elements[0]), f"[{100*step/len(elements[0]):4.2f}%]",
              " elements read:", f"{t1:6.4f} s/{Nstep} elems")
        t1 = time()
        
    a,b,c,d = k[0], k[1], k[2], k[3]
    
    if (sp_basis[a].i_sh > sp_basis[b].i_sh): continue
    if (sp_basis[a].i_sh > sp_basis[c].i_sh): continue
    if (sp_basis[c].i_sh > sp_basis[d].i_sh): continue
    
    ta,tb,tc,td = sp_basis[a].mt, sp_basis[b].mt, sp_basis[c].mt, sp_basis[d].mt
    tt = ta + tb + tc + td
    if (ta > tb) or (tc > td): continue # exclude all non pnpn elements
    
    
    abcd_sh_key = (sp_basis[a].i_sh, sp_basis[b].i_sh, 
                   sp_basis[c].i_sh, sp_basis[d].i_sh)
    
    ja,jb,jc,jd = sp_basis[a].j, sp_basis[b].j, sp_basis[c].j, sp_basis[d].j
    ma,mb,mc,md = sp_basis[a].mj, sp_basis[b].mj, sp_basis[c].mj, sp_basis[d].mj
    
    Mbra = (ma + mb) // 2
    Mket = (mc + md) // 2
    
    Jb_min = abs(ja - jb) // 2
    Jb_max =    (ja + jb) // 2
    Jk_min = abs(jc - jd) // 2
    Jk_max =    (jc + jd) // 2
        
    for Jbra in range(Jb_min, Jb_max+1):
        if (abs(Mbra) > Jbra): continue
        cgc1 = float(CG(S(ja)/2,S(ma)/2, S(jb)/2,S(mb)/2, S(Jbra),S(Mbra))
                       .doit())
        
        for Jket in range(Jk_min, Jk_max+1):
            if (abs(Mket) > Jket): continue
            cgc2 = float(CG(S(jc)/2,S(mc)/2, S(jd)/2,S(md)/2, S(Jket),S(Mket))
                           .doit())
            
            aux_val = cgc1 * cgc2  * vdd
            
            jkey = (Jbra, Mbra, Jket, Mket)

            if abcd_sh_key in hamilJM[tt]:
                if jkey in hamilJM[tt][abcd_sh_key]:
                    hamilJM[tt][abcd_sh_key][jkey] += aux_val
                else:
                    hamilJM[tt][abcd_sh_key][jkey] = aux_val
            else:
                hamilJM[tt][abcd_sh_key] = {jkey : aux_val}
            
            # if abcd_sh_key == (1,1,1,1):
            #     if (Jbra != Jket): continue
            #     if (Jbra == 0): continue
            #     print(f"{cgc1:+7.5f}*{cgc1:+7.5f}*{vdd:+7.5f}={aux_val:+7.5f}",
            #           f" sum={hamilJM[tt][abcd_sh_key][jkey]:7.5f}  sp=",
            #           k," jkey=", jkey)
                

## normalize with the (for npnp states has no importance)
TENSOR_ORD = 2
#2 prepare to export the elements
HOsh_dim = max(shell_ofSP.values())
print(" CONSTR RED HAMIL ")

def constructRedHamil(Jb_min, Jb_max, Jk_min, Jk_max, abcd_key):
    global TENSOR_ORD
    global hamilJM
        
    all_zero = dict([(i, True) for i in range(TENSOR_ORD+1)])
    hamilRed = {-2: {}, 0: {}, 2: {}}
    
    d_ab, d_cd = (abcd_key[0] == abcd_key[1])*1, (abcd_key[2] == abcd_key[3])*1
    
    
    for Jbra in range(Jb_min, Jb_max+1):
        for Mbra in range(-Jbra, Jbra+1):
            for Jket in range(Jk_min, Jk_max+1):
                norm = np.sqrt((1 + d_ab*((-1)**Jbra))*(1 + d_cd*((-1)**Jket)))
                norm /= ((1+ d_ab)*(1 + d_cd))
                
                KKmin = abs(Jbra - Jket)
                KKmax = min(Jbra + Jket, TENSOR_ORD)
                if (KKmin > TENSOR_ORD): continue
                
                jredkey = (Jbra, Jket)
                
                for Mket in range(-Jket, Jket+1):
                    
                    MM = Mbra - Mket
                    
                    for KK in range(KKmin, KKmax+1):
                        if abs(MM) > KK: continue
                        
                        cg1 = CG(S(Jbra),-S(Mbra), S(KK),S(MM), S(Jket),-S(Mket))
                        cg1 = float(cg1.doit())
                        if abs(cg1) < 1.0e-12: continue
                        
                        ## PRocedure for summing M,M'mu (ingluding the addtional factor [Jbra] from WE)
                        ##
                        # aux_val  = np.sqrt((2*Jket + 1)*(2*Jbra + 1)) / cg1
                        # aux_val /= ((2*Jbra + 1)*(2*Jket + 1)*(2*KK + 1))
                        # aux_val *= ((-1)**(Mbra+KK+Mket)) 
                        
                        ## Procedure only one element (not sum)
                        ##
                        aux_val = np.sqrt((2*Jket + 1)/(2*Jbra + 1)) / cg1
                        aux_val *= ((-1)**(Mbra+KK+Mket))
                        
                        jkey = (Jbra, Mbra, Jket, Mket)
                        
                        for tt in (-2, 0, 2):
                            if abcd_key not in hamilJM[tt]: continue
                            
                            vJM = hamilJM[tt][abcd_key].get(jkey)
                            if vJM == None: continue
                            
                            if tt != 0: 
                                vJM = vJM * norm
                            aux_val2 = vJM * aux_val
                            
                            if KK in hamilRed[tt]:
                                if jredkey in hamilRed[tt]:
                                    
                                    if abs(aux_val2 - hamilRed[tt][KK][jredkey]) > 1.0e-5:
                                        # if abcd_key != (1,1,1,1): continue
                                        elem = hamilRed[tt][KK][jredkey]
                                        print(f" error, value [{aux_val2:+7.5f}] /= [{elem:+7.5f}]",
                                              "jkey=",jkey, "KKMM",KK,MM )
                                    
                                    # hamilRed[tt][jredkey]  += aux_val2 #+=
                                else:
                                    hamilRed[tt][KK][jredkey]  = aux_val2
                            else:
                                hamilRed[tt][KK]  = {jredkey: aux_val2}
                            
                            # if abcd_key == (1,1,1,1):
                            #     print(f"{cgc1:+7.5f} auxV={aux_val:+7.5f}  Jkey",
                            #           jkey, "KKMM",KK,MM,"vJM=",vJM,"*auxV=",
                            #           aux_val2," sum=",hamilRed[tt][KK][jredkey])
                            
                            
                            if abs(aux_val2) > 1.0e-15:
                                all_zero[KK] *= False
    
    ## 
    
    return all_zero, hamilRed

out_texts = dict([(KK, '') for KK in range(TENSOR_ORD+1)])
for a in range(1, HOsh_dim+1):
    for b in range(a, HOsh_dim+1):
        ja, jb = sh_basis[a].j, sh_basis[b].j 
        
        
        for c in range(a, HOsh_dim+1):
            for d in range(c, HOsh_dim+1):
                jc, jd = sh_basis[c].j, sh_basis[d].j
                
                Jb_min = abs(ja - jb) // 2
                Jb_max =    (ja + jb) // 2
                Jk_min = abs(jc - jd) // 2
                Jk_max =    (jc + jd) // 2
                Js = [Jb_min, Jb_max, Jk_min, Jk_max]
                
                abcd_key = (a,b,c,d)
                header = ' 0 5  '+ ' '.join([sh_basis[a].ind_ant, sh_basis[b].ind_ant,
                                             sh_basis[c].ind_ant, sh_basis[d].ind_ant])
                
                all_zero, hamilred = constructRedHamil(*Js, abcd_key)
                
                if not False in all_zero.keys(): continue
                
                ## Headers for the FILES.
                for KK in range(TENSOR_ORD+1):
                    if all_zero[KK]: continue
                    
                    if KK > 0: # tensor files
                        J3 = max(Jb_min , Jk_min)
                        J2 = min(Jb_max , Jk_max)
                        J1 = max(J3 - KK, min(Jb_min, Jk_min))
                        J4 = min(J2 + KK, max(Jb_max, Jk_max))
                        
                        aux = [str(j) for j in [J1,J2,J3,J4]]
                        out_texts[KK] += header+'  '+ ' '.join(aux)+'\n'
                    else: # scalar file
                        jmin = max(Jb_min, Jk_min)
                        jmax = min(Jb_max, Jk_max)
                        aux = [str(j) for j in [jmin,jmax]]
                        out_texts[KK] += header+'  '+ ' '.join(aux)+'\n'
                
                ## Write in the files
                for Jbra in range(Jb_min, Jb_max+1):
                    for Jket in range(Jk_min, Jk_max+1):
                        KKmin = abs(Jbra - Jket)
                        KKmax = min(Jbra + Jket, TENSOR_ORD)
                        if (KKmin > TENSOR_ORD) : continue
                        
                        for KK in range(KKmin, KKmax+1):
                            if all_zero[KK]: continue
                            
                            if KK > 0: # in Tensor file, put a J,J' before
                                out_texts[KK] += '{:4} {:4}'.format(Jbra,Jket)
                            
                            for tt in (-2, 0, 2):
                                if KK in hamilred[tt]:
                                    aux_val = hamilred[tt][KK].get((Jbra,Jket), 0.0)
                                    out_texts[KK] += '{:15.10f}'.format(aux_val)
                                else:
                                    out_texts[KK] += '{:15.10f}'.format(0.0)
                            out_texts[KK] += '\n'

## write the files
kwargs = dict([(KK, FOLDER+'onlyDD_k{}.txt'.format(KK)) 
                                               for KK in range(TENSOR_ORD+1)])
FILESOUTPUT = {
    0 : 'onlyDD_scalar.txt', **kwargs}
for KK in range(TENSOR_ORD+1):
    with open(FILESOUTPUT[KK], 'w+') as f:
        f.write(out_texts[KK])


#%% EXAMPLE OF TR Reconstruction/ PERMUTATION for taurus PROCESS (and complete comparison)
## // SINGLE PARTICLE INDEX (i_sp, i_sh, n,l,2j,2m, 2mt, i_sp_TR)
# basis = """
#    1   1   0   0   1   1  -1   2
#    2   1   0   0   1  -1  -1   1
#    3   2   0   1   1   1  -1   4
#    4   2   0   1   1  -1  -1   3
#    5   3   0   1   3   3  -1   8
#    6   3   0   1   3   1  -1   7
#    7   3   0   1   3  -1  -1   6
#    8   3   0   1   3  -3  -1   5
#    9   1   0   0   1   1   1  10
#   10   1   0   0   1  -1   1   9
#   11   2   0   1   1   1   1  12
#   12   2   0   1   1  -1   1  11
#   13   3   0   1   3   3   1  16
#   14   3   0   1   3   1   1  15
#   15   3   0   1   3  -1   1  14
#   16   3   0   1   3  -3   1  13
#   """
basis = """
   1   1   0   0   1   1  -1   2
   2   1   0   0   1  -1  -1   1
   3   2   0   1   1   1  -1   4
   4   2   0   1   1  -1  -1   3
   5   1   0   0   1   1   1   6
   6   1   0   0   1  -1   1   5
   7   2   0   1   1   1   1   8
   8   2   0   1   1  -1   1   7
  """
BREAK_PAR = True
NEW_COUNTING = True

sp_basis = {}
    
for index, line in enumerate(basis.split('\n')[1:-1]):
    line = line.strip()
    # print(line.split(separator))
    i_sp, i_sh, n,l,j,m, mt,i_tr = line.split('  ')
    
    i_sp, i_sh, i_tr = int(i_sp), int(i_sh), int(i_tr)
    n,l,j,m, mt      = int(n),int(l),int(j),int(m), int(mt)
    sp_basis[i_sp]   = SP(i_sp, n,l,j,m, mt, i_tr, i_sh)
    shell_ofSP[i_sp] = i_sh
        
# getBasisSP(basis, separator='  ')

N = len(sp_basis)

full_hamiltonian = []
for a in range(N):
    full_hamiltonian.append([])
    ta = sp_basis[a+1].mt
    ma = sp_basis[a+1].mj
    la = sp_basis[a+1].l
    for b in range(N):
        full_hamiltonian[a].append([])
        tb = sp_basis[b+1].mt
        mb = sp_basis[b+1].mj
        lb = sp_basis[b+1].l
        for c in range(N):
            full_hamiltonian[a][b].append([])
            tc = sp_basis[c+1].mt
            mc = sp_basis[c+1].mj
            lc = sp_basis[c+1].l
            for d in range(N):
                td = sp_basis[d+1].mt
                md = sp_basis[d+1].mj
                ld = sp_basis[d+1].l
                
                st = None if ta+tb != tc+td else 0
                if st != None:
                    st = None if (ma+mb-mc-md) != 0 else 0
                if (not BREAK_PAR) and st != None :
                    st = None if ((la+lb+lc+ld) % 2 == 1) else 0
                
                full_hamiltonian[a][b][c].append(st)

hamil_DD_abcd = []


def step_reconstruct_2BTR(a_sp, b_sp, c_sp, d_sp):
    
    ta, tb = sp_basis[a_sp].i_tr, sp_basis[b_sp].i_tr
    tc, td = sp_basis[c_sp].i_tr, sp_basis[d_sp].i_tr
    
    perm  = 0
    phase = (-1)**(sum([sp_basis[st].j for st in (a_sp,b_sp,c_sp,d_sp)])/2)
    
    if ( ta > tb ):
        tmp = ta
        ta = tb
        tb = tmp
        phase = -phase
        perm = perm + 1
    
    if ( tc > td ):
        tmp = tc
        tc = td
        td = tmp
        phase = -phase
        perm = perm + 2
    
    if ( (ta > tc) or ((ta == tc) and (tb > td)) ):
        perm = perm + 4
        
    # if all equal, perm = 0
    
    if ( phase < 0 ):
        perm = perm - 8
    
    return perm



## 1. read the matrix elements for the hamil_DD to get the matrix elems as in taurus
count_nonnull, count_step = 0, 0
if NEW_COUNTING:
    for a in range(N):
        # for b in range(a+1, N):
        for b in range(a+1, N):
            a_sp, b_sp = a+1, b+1
            
            ta, tb = sp_basis[a_sp].mt, sp_basis[b_sp].mt
            ma, mb = sp_basis[a_sp].mj, sp_basis[b_sp].mj
            la, lb = sp_basis[a_sp].l,  sp_basis[b_sp].l
            
            if (ma + mb < 0):continue
            
            # for c in range(a, N):
            for c in range(a, N):
                
                d_max = b+1 if c==a else N  ## note that range limit is +1 the last number
                
                for d in range(c+1, d_max):
                    c_sp, d_sp = c+1, d+1
                    
                    tc, td = sp_basis[c_sp].mt, sp_basis[d_sp].mt
                    mc, md = sp_basis[c_sp].mj, sp_basis[d_sp].mj
                    lc, ld = sp_basis[c_sp].l,  sp_basis[d_sp].l
                    
                    count_step += 1
                    if ta + tb != tc + td: continue
                    if ma + mb - mc - md  != 0: continue
                    
                    if not BREAK_PAR:
                        if (la + lb + lc + ld)% 2 == 1: continue
    
                    perm = step_reconstruct_2BTR(a_sp, b_sp, c_sp, d_sp)
                    hamil_DD_abcd.append([a_sp, b_sp, c_sp, d_sp, perm])
                    count_nonnull += 1

else:
    ## Same process with the old counting method (from the article)
    print(" OLD COUNING OF STATES")
    for a in range(N):
        # for b in range(a+1, N):
        for c in range(a, N):
            a_sp, c_sp = a+1, c+1
            
            ta, tc = sp_basis[a_sp].mt, sp_basis[c_sp].mt
            ma, mc = sp_basis[a_sp].mj, sp_basis[c_sp].mj
            la, lc = sp_basis[a_sp].l,  sp_basis[c_sp].l
                    
            # for c in range(a, N):
            for d in range(c, N):
                ## note that range limit is +1 the last number
                b_max = d+1 if c==a else N 
                # for d in range(c+1, d_max):
                # b_max = N
                for b in range(a, b_max):
                    b_sp, d_sp = b+1, d+1
                    
                    tb, td = sp_basis[b_sp].mt, sp_basis[d_sp].mt
                    mb, md = sp_basis[b_sp].mj, sp_basis[d_sp].mj
                    lb, ld = sp_basis[b_sp].l,  sp_basis[d_sp].l
                    
                    count_step += 1
                    if ta + tb != tc + td: continue
                    if ma + mb - mc - md  != 0: continue
                    if (ma + mb < 0):continue
                    
                    if not BREAK_PAR:
                        if (la + lb + lc + ld)% 2 == 1: continue
                    # hamil_DD[a][b][c][d] += 1
                    perm = step_reconstruct_2BTR(a_sp, b_sp, c_sp, d_sp)
                    hamil_DD_abcd.append([a_sp, b_sp, c_sp, d_sp, perm])
                    count_nonnull += 1
                

print(f"\n *** Reading as in subroutine [calculate_densityDep_hamiltonian] ***")
print(f"valid elements(hamil_DD_abcd) {len(hamil_DD_abcd)} (Break Parity [{BREAK_PAR}])")
print(f"sp_dim={N}: Count max m.e.[{N**4}]  steps[{count_step}] valid[{count_nonnull}]")

def find_timerev(p, a_sp,b_sp,c_sp,d_sp):
    
    ta, tb = sp_basis[a_sp].i_tr, sp_basis[b_sp].i_tr
    tc, td = sp_basis[c_sp].i_tr, sp_basis[d_sp].i_tr
    
    if   p in (-8,0):
        a_sp,b_sp,c_sp,d_sp = ta,tb,tc,td
    elif p in (-7,1):
        a_sp,b_sp,c_sp,d_sp = tb,ta,tc,td
    elif p in (-6,2):
        a_sp,b_sp,c_sp,d_sp = ta,tb,td,tc
    elif p in (-5,3):
        a_sp,b_sp,c_sp,d_sp = tb,ta,td,tc
    
    elif p in (-4,4):
        a_sp,b_sp,c_sp,d_sp = tc,td,ta,tb
    elif p in (-3,5):
        a_sp,b_sp,c_sp,d_sp = tc,td,tb,ta
    elif p in (-2,6):
        a_sp,b_sp,c_sp,d_sp = td,tc,ta,tb
    elif p in (-1,7):
        a_sp,b_sp,c_sp,d_sp = td,tc,tb,ta
    else:
        raise Exception(f"p={p} !!!")
    return a_sp,b_sp,c_sp,d_sp
        

DOUBLE_COUNT = 0
NULL_COUNT   = 0
NOT_COUNT    = 0
double_counted_states = {}
not_counted  = set()
counted      = set()
invalid_mes  = set()

## 2. Build the Fields, but counting the proceeding m.e. of the permutations
def append_perm_2hamil(a,b,c,d, perm_tr_str):
    global DOUBLE_COUNT, NULL_COUNT,double_counted_states
    
    if   full_hamiltonian[a][b][c][d] == None:
        NULL_COUNT += 1 
    elif full_hamiltonian[a][b][c][d] > 0:
        str_ = f"{a+1},{b+1},{c+1},{d+1}"
        
        if perm_tr_str not in double_counted_states:
            double_counted_states[perm_tr_str] = [str_,]
        else:
            double_counted_states[perm_tr_str].append(str_)
        DOUBLE_COUNT += 1 
    else:
        full_hamiltonian[a][b][c][d] += 1
    
## count the permutations 
tr_vals = ('F', 'T')
for a_sp, b_sp, c_sp, d_sp, perm in hamil_DD_abcd:
    
    for tr in range(2):
        tr_i = tr_vals[tr]
        if tr_i == 'T':
            if sp_basis[a_sp].mj + sp_basis[b_sp].mj == 0: 
                continue
            a_sp, b_sp, c_sp, d_sp = find_timerev(perm, a_sp, b_sp, c_sp, d_sp)
        
        a, b, c, d = a_sp-1, b_sp-1, c_sp-1, d_sp-1
        
        str_unperm = f"{a_sp},{b_sp},{c_sp},{d_sp} tr={tr_i}"
        
        if NEW_COUNTING: # False: #
            # full_hamiltonian[a][b][c][d] += 1
            append_perm_2hamil(a,b,c,d, str_unperm)
            
            if (a == b) and (c == d):
                if (a == c): continue
                # full_hamiltonian[c][d][a][b] += 1
                append_perm_2hamil(c,d,a,b, str_unperm)
                # # full_hamiltonian[c][d][b][a] += 1
                # append_perm_2hamil(c,d,b,a, str_unperm)
                # # full_hamiltonian[d][c][a][b] += 1
                # append_perm_2hamil(d,c,a,b, str_unperm)
                # # full_hamiltonian[d][c][b][a] += 1
                # append_perm_2hamil(d,c,b,a, str_unperm)
                continue
            # full_hamiltonian[a][b][d][c] += 1
            append_perm_2hamil(a,b,d,c, str_unperm)
            # full_hamiltonian[b][a][c][d] += 1
            append_perm_2hamil(b,a,c,d, str_unperm)
            # full_hamiltonian[b][a][d][c] += 1
            append_perm_2hamil(b,a,d,c, str_unperm)
            
            if (a == c) and (b == d): continue  ## doesn't count the f2b
            
            # full_hamiltonian[c][d][a][b] += 1
            append_perm_2hamil(c,d,a,b, str_unperm)
            # full_hamiltonian[c][d][b][a] += 1
            append_perm_2hamil(c,d,b,a, str_unperm)
            # full_hamiltonian[d][c][a][b] += 1
            append_perm_2hamil(d,c,a,b, str_unperm)
            # full_hamiltonian[d][c][b][a] += 1
            append_perm_2hamil(d,c,b,a, str_unperm)
        else:
            # [a][b][c][d]
            append_perm_2hamil(a,b,c,d, str_unperm)
            if (c != d):
                # [a][b][d][c]
                append_perm_2hamil(a,b,d,c, str_unperm)
            if (b != a):
                # [b][a][c][d]
                append_perm_2hamil(b,a,c,d, str_unperm)
            if (b != a) and (c != d):
                # [b][a][d][c]
                append_perm_2hamil(b,a,d,c, str_unperm)
            if (a != c) or (b != d):
                # [c][d][a][b]
                append_perm_2hamil(c,d,a,b, str_unperm)
                if (a != b): 
                    # [c][d][b][a]
                    append_perm_2hamil(c,d,b,a, str_unperm)
                if (c != d):
                    # [d][c][a][b]
                    append_perm_2hamil(d,c,a,b, str_unperm)
                if (a != b) and (c != d):
                    # [d][c][b][a]
                    append_perm_2hamil(d,c,b,a, str_unperm)


## See m.e. not counted
not_counted_notPermut = dict()
def save_base_elem(a_sp, b_sp, c_sp, d_sp):
    global not_counted_notPermut
    
    ab = sorted((a_sp, b_sp))
    cd = sorted((c_sp, d_sp))
    if ab >= cd:
        key_ = tuple([*ab, *cd])
        if key_ in not_counted_notPermut:
            not_counted_notPermut[key_] += 1
        else: 
            not_counted_notPermut[key_] = 1
    else:
        key_ = tuple([*cd, *ab])
        if key_ in not_counted_notPermut:
            not_counted_notPermut[key_] += 1
        else: not_counted_notPermut[key_] = 1
    

for a in range(N):
    for b in range(N):
        for c in range(N):
            for d in range(N):
                me_spstates = (a+1,b+1,c+1,d+1)
                
                if full_hamiltonian[a][b][c][d] == None:
                    invalid_mes.add(me_spstates)
                elif   full_hamiltonian[a][b][c][d] == 0:                    
                    not_counted.add(me_spstates)
                    save_base_elem(*me_spstates)
                elif full_hamiltonian[a][b][c][d] == 1:
                    counted.add(me_spstates)

NOT_COUNT = len(not_counted)
COUNTED = len(counted)
INVALID = len(invalid_mes)
print("\n *** RESULTS *** -----------------------")
print(f"Null counted  [{NULL_COUNT}]  (elements that break isospin and total mj uneven)")
print(f"double counted[{DOUBLE_COUNT}]")
for st, perms in  double_counted_states.items():
    vals  = [str(sp_basis[int(sp)]) for sp in st.split(' ')[0].split(',')]
    print(" dc:{}[{}]\t({:13}{:13} | {:13}{:13})".format(st, len(perms), *vals))
print()
print(f"Not Counted[{NOT_COUNT}] Counted[{COUNTED}+{DOUBLE_COUNT}] Invalid[{INVALID}]")
print(f"TOTAL [{COUNTED+DOUBLE_COUNT+NOT_COUNT+INVALID}] sp_dim^4={N**4}")
if NOT_COUNT != 0:
    for i, sts in enumerate(sorted(not_counted_notPermut.items(), key=lambda x: x[0])):
        vals  = [str(sp_basis[int(sp)]) for sp in sts[0]]
        counts_ = sts[1]
        print(" nc{:2}:{}[{}] \t({:13}{:13} | {:13}{:13})".format(i,*sts,*vals))




#%% Convert CKI and USBD into J scheme

FOLDER = 'FieldAndME_tests/ddMatrixElementsExportable16O_sph/'
fileImport = 'cki'
fileImport = 'usdb'
# fileImport = 'kb3g.a42'
jt_elems = {}
header_file = ''

fmt_impotFile = '{:8.4f}  '
if fileImport == 'kb3g.a42':
    fmt_impotFile = ' {:8.3f}  '


with open(FOLDER+fileImport, 'r') as f:
    data = f.readlines()
    
    for line, vals in enumerate(data):
        if line < 4:
            if line == 1:
                sh_vals  = [int(val) for val in vals.split()]
                sh_vals2 = []
                for v in sh_vals:
                    if v > 1000:
                        v = 10000*(v // 1000) + (v % 1000)
                    sh_vals2.append(str(v))
                vals = ' ' + ' '.join(sh_vals2) + '\n'                    
                    
            header_file += vals
            continue
        
        vals = vals.strip()
        
        if vals.startswith('0 1 '):
            vals = vals.split()
            jmin, jmax = int(vals[-2]), int(vals[-1])
            
            a,b,c,d = int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5])
            
            ## Convert to L great than 10 format ****
            fix_lgt10 = [a,b,c,d]
            sh_key = []
            for esh in fix_lgt10:
                if esh > 1000:
                    n, lj = esh // 1000, esh % 1000
                    #l, j  = lj // 100, lj % 100
                    esh = 10000*n + lj           
                sh_key.append(esh)
            sh_key = tuple(sh_key)
            
            jt_elems[sh_key] = {0: {}, 1: {}, 'jmin':jmin, 'jmax':jmax}
            T=0
        else:
            vals = vals.split()
            
            for j, val in  enumerate(vals):
                val = float(val)
                jt_elems[sh_key][T][jmin+j] = val
                
                assert j + jmin <= jmax
            
            T += 1

## suhonen formula for reconstruction
def suhonen_formula(a,b,c,d,J, v_T0, v_T1, tt):
    d_ab = int(a==b)
    d_cd = int(c==d)
    ## tt= pppp pnpn pnnp nppn npnp nnnn
    if   tt in (0, 5):
        return v_T1
    elif tt in (1, 4): # pnpn npnp
        return 0.5*((v_T1*(( (1 + (-1*d_ab)**J) * (1 + (-1*d_cd)**J) )**0.5)) +
                    (v_T0*(( (1 - (-1*d_ab)**J) * (1 - (-1*d_cd)**J) )**0.5)))
    elif tt in (2, 3): # pnpn npnp
        return 0.5*((v_T1*(( (1 + (-1*d_ab)**J) * (1 + (-1*d_cd)**J) )**0.5)) -
                    (v_T0*(( (1 - (-1*d_ab)**J) * (1 - (-1*d_cd)**J) )**0.5)))
    else:
        raise Exception(" invalid tt")

j_elems = {}
j_text = header_file
for sh_key in jt_elems: 
    j_elems[sh_key] = {}
    
    jmin, jmax = jt_elems[sh_key]['jmin'], jt_elems[sh_key]['jmax']
    
    j_text += ' 0 5 '+' '.join([str(v) for v in sh_key])+f' {jmin} {jmax}\n' 
    for J in range(jmin, jmax+1):
        vT0 = jt_elems[sh_key][0][J]
        vT1 = jt_elems[sh_key][1][J]
        
        j_elems[sh_key][J] = []
        aux = ''
        for tt in range(6):
            j_elems[sh_key][J].append(suhonen_formula(*sh_key,J,vT0,vT1,tt))
            aux += fmt_impotFile.format(j_elems[sh_key][J][tt])
        aux += '\n'
        if j_elems[sh_key][J][0] < 0:
            j_text += '  ' + aux.strip() + '\n'
        else:
            j_text += '   ' + aux.strip() + '\n'

print(j_text)
with open(FOLDER+fileImport+'.2b', 'w+') as f:
    f.write(j_text)
