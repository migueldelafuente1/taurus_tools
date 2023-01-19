# -*- coding: utf-8 -*-
"""
Created on Sat May 21 20:16:52 2022

@author: Miguel
"""

filename = 'uncoupled_DD.2b'
filename = 'uncoupled_DD_O16MZ2.2b'
# filename = 'uncoupled_DD_O16MZ3.2b'
filename = 'uncoupled_DD_0001.2b'

with open(filename, 'r') as f:
    data = f.read()
    _, header, data = data.split('//')
    header      = header.split('\n')
    index_temp  = header[0]
    header  = header[1:-1]
    data    = data.split('\n')[1:-1]
    
#%% IMPORT: get the single particle states for each index.
# SINGLE PARTICLE INDEX (i_sp, i_sh, n,l,2j,2m, 2mt, i_sp_TR)

class SpState():
    
    def __init__(self, sp_tuple):
        
        if False in [isinstance(i, int) for i in sp_tuple]:
            sp_tuple = tuple([int(v) for v in sp_tuple])
        
        self.sp_index = sp_tuple[0]
        self.sh_state = sp_tuple[1]
        
        self.n  = sp_tuple[2]
        self.l  = sp_tuple[3]
        self.j  = sp_tuple[4]
        self.m  = sp_tuple[5]
        
        self._antoineIndex = self.getAntoineIndex()
        
        self.mt = sp_tuple[6]
        self.proton_state  = self.mt == -1
        self.neutron_state = not self.proton_state
        
        self.tr_sp_index = sp_tuple[7]
        
    def isSameJstate(self, spState2, considerMt=True):
        """Check if the states correspond to the same state n,l,j
            and also particle-label by default """
        if ((spState2.tr_sp_index == self.sp_index) 
            or (spState2.sp_index == self.sp_index)):
            return True
        
        if considerMt and spState2.mt != self.mt:
            return False
        
        # if spState2.n != self.n or spState2.l != self.l:
        if self._antoineIndex == spState2._antoineIndex:
            return False
        
        if spState2.j == self.j:
            return True
        return  False
    
    def getAntoineIndex(self, le_g_10=True):
        if le_g_10:
            return 10000*self.n + 100*self.l + self.j
        return 1000*self.n + 100*self.l + self.j
    
    def __str__(self):
        # return "{:03}".format(self._antoineIndex) \
        return  str(self._antoineIndex) \
                +"{}({:+2})".format('p' if self.proton_state else 'n', self.m)
        

print(index_temp)
sp_dict = {}
sh_dict = {}
sh_2j   = {}
for sp_line in header:
    elems = sp_line.strip().split(',')
    i_sp = elems[0]
    i_sp = int(i_sp)
    
    # elems = tuple([int(v) for v in elems])
    elem = SpState(elems)
    sp_dict[i_sp] = elem
    
    ant_ind = elem.getAntoineIndex()
    if ant_ind not in sh_dict:
        sh_dict[ant_ind] = {i_sp : elem}
        sh_2j  [ant_ind] = elem.j
    else:
        sh_dict[ant_ind][i_sp] = elem

sp_dim = max(sp_dict.keys()) // 2


#%% get the matrix element list.
matrix_elements = {}
for me_line in data:
    # print(me_line)
    a, b, c, d, me = me_line.strip().split()
    a, b, c, d = int(a), int(b), int(c), int(d)
    print(" ".join((str(sp_dict[x]).rjust(11) for x in (a,b,c,d))) 
          +" : "+str(me))
    
    matrix_elements[((a, b),(c, d))] = float(me)




#%% recouple matrx elements function

from copy import deepcopy

from sympy import S
from sympy.physics.quantum.cg import CG


# TODO obtain explicitly the Recoupling me for a single m.e. 
alpha = 1
beta  = -alpha
gamma = 1
delta = -gamma
cg = CG(S(1)/2, S(alpha)/2, S(1)/2, S(beta)/2, 1, 0)
cg.doit()

# def recouple_J:
#     pass
def N_coeff_J(a, b, J):
    
    if (a.mt==b.mt) and (a.n==b.n) and (a.l==b.l) and (a.j==b.j):
        if J % 2 == 1:
            return 0.0, True
        else:
            return 0.7071067811865475, False
    else:
        return 1.0, False
    
def _print(*args):
    print(*args)

def get_recoupling(a_sh,b_sh,c_sh,d_sh, J, M=0, exchange_braket=True):
    PRINT = False
    MM = 2*M
    
    a_sp_states = deepcopy(sh_dict[a_sh])
    # if a_sp_states is sh_dict[b_sh]:
    #     b_sp_states = deepcopy(a_sp_states)
    # else:
    #     b_sp_states = sh_dict[b_sh]
    b_sp_states = deepcopy(sh_dict[b_sh])
        
    c_sp_states = deepcopy(sh_dict[c_sh])
    # if c_sp_states is sh_dict[d_sh]:
    #     d_sp_states = deepcopy(c_sp_states)
    # else:
    #     d_sp_states = sh_dict[d_sh]
    d_sp_states = deepcopy(sh_dict[d_sh])
    
    
    v_pppp = 0.0
    v_pnpn = 0.0
    v_nnnn = 0.0
    
    Neut_sp_dim = max(sp_dict.keys()) // 2 + 1
    
    for a_sp, a in a_sp_states.items():
        if PRINT: _print()
        if a_sp >= Neut_sp_dim:
            continue
        for b_sp, b in b_sp_states.items():
            if PRINT: _print("a:{} b:{}".format(a_sp, b_sp))
            if b_sp <  Neut_sp_dim:
                continue
            # if b_sp < a_sp:
            #     continue
            if a.m + b.m != MM:
                continue
            NabJ, null = N_coeff_J(a, b, J)
            if null:
                # print("Null ab")
                continue
            
            key_bra = (a_sp, b_sp)  ##(min(a_sp, b_sp), max(a_sp, b_sp))
            
            for c_sp, c in c_sp_states.items():
                if c_sp >= Neut_sp_dim: continue
                
                for d_sp, d in d_sp_states.items():
                    if PRINT: _print("    c:{} d:{}".format(c_sp, d_sp))
                    if d_sp < Neut_sp_dim: continue
                    # if d_sp <  c_sp :
                    #     continue
                    if d.m + c.m != MM:
                        continue
                    NcdJ, null = N_coeff_J(c, d, J)
                    if null:
                        # print("Null cd")
                        continue
                    key_ket = (c_sp, d_sp)
                    
                    me  = matrix_elements.get((key_bra, key_ket), None)
                    tr_ab = (a.tr_sp_index, b.tr_sp_index)
                    tr_cd = (c.tr_sp_index, d.tr_sp_index)
                    mtr   = matrix_elements.get((tr_ab, tr_cd), None)
                    # me_dc = None
                    # if c_sh < d_sh:
                    me_dc = matrix_elements.get((key_bra, (d_sp, c_sp)), None)
                    
                    if not me:
                        if PRINT: _print("not me")
                        continue
                    
                    # build the m.e. from the hamilt. symmerty
                    # if the tr m.e is present, the loop will sum it later.
                    mtr   = 0.0 if mtr else me
                    me_dc = 0.0 if not me_dc else me_dc#*(-1)**(J+(c.j + d.j)/2)
                    
                    cg1 = float(CG(S(a.j)/2,S(a.m)/2, S(b.j)/2,S(b.m)/2, J, M).doit())
                    cg2 = float(CG(S(c.j)/2,S(c.m)/2, S(d.j)/2,S(d.m)/2, J, M).doit())
                    
                    aux = NabJ * NcdJ * cg1 * cg2 * (me + mtr + me_dc)
                    details = (str(a) for a in [NabJ, NcdJ, cg1, cg2])
                    
                    
                    t = sum([sp.mt for sp in (a,b,c,d)])
                    if   (t== 0):
                        v_pnpn += aux
                    elif (t== 4):
                        v_nnnn += aux
                    elif (t==-4):
                        v_pppp += aux
                    else:
                        raise Exception("Invalid particle State t={:t}".format(t))
                    
                    if PRINT: _print(key_bra, key_ket, aux, "details: [", ",".join(details), "]")
    
    # run the exchanged braket 
    if exchange_braket:
        # v_pp2, v_pn2, v_nn2 = get_recoupling(c_sh,d_sh,a_sh,b_sh, J, M=0, 
        #                                       exchange_braket=False)
        # v_pppp += v_pp2
        # v_pnpn += v_pn2
        # v_nnnn += v_nn2
        pass
        
    return v_pppp, v_pnpn, v_nnnn



sts  = (1, 101, 1, 103)
JM   = (1, 0)
print("FINAL <",*sts,"> JM(",*JM,")=",get_recoupling(*sts, *JM))

kets = [(1,24), (2,23), (23,2), (24,1)]
kets = kets + [(x[1],x[0]) for x in kets]
bras = [(1,27), (2,26), (21,7), (22,6)]
bras = bras + [(x[1],x[0]) for x in bras]

# for b in bras:
#     for k in kets:
#         print("<{}|v|{}> {:8.5f}    <{}|v|{}> {:8.5f}"
#               .format(b, k, matrix_elements.get((b,k), 0.0),
#                       k, b, matrix_elements.get((k,b), 0.0))
#               )
        


#%% Read the valence space to recouple each SHO TBME state 

combinations_sh_without_repetition = []
sh_list = list(sh_dict.keys())
for k1 in range(len(sh_list)):
    for k2 in range(k1, len(sh_list)):
        combinations_sh_without_repetition.append((sh_list[k1], sh_list[k2]))

def get_J_range(a_sh, b_sh, sh_dict):
    
    ja = sh_dict[a_sh][list(sh_dict[a_sh].keys())[0]].j
    jb = sh_dict[b_sh][list(sh_dict[b_sh].keys())[0]].j
    
    J_min = abs(ja - jb) // 2
    J_max = abs(ja + jb) // 2
    
    return [j for j in range(J_min, J_max+1)]

def get_str_sp(st):
    if st == 1:
        return "001"
    return str(st)

i = 0
matrix_elements_J = {}
order_meJ = []
for i_br, bra in enumerate(combinations_sh_without_repetition):
    J_bra_range = get_J_range(*bra, sh_dict)
    
    for i_kt in range(i_br, len(combinations_sh_without_repetition)):
        ket = combinations_sh_without_repetition[i_kt]
        J_ket_range = get_J_range(*ket, sh_dict)
        
        J_range = list(filter(lambda j: j in J_ket_range, J_bra_range))
        
        i+=1
        J_block = {}
        all_null = True
        for J in J_range:
            pass
            # print("[",i, "]", bra, ket, " J:", J)
            v_pp, v_pn, v_nn = get_recoupling(*bra, *ket, J)
            if any(abs(x)>1.e-10 for x in (v_pp, v_pn, v_nn)):
                J_block[J] = [v_pp, v_pn, -v_pn, -v_pn, v_pn, v_nn]
                all_null = False
            else:
                J_block[J] = [0.0]*6
        
        if not all_null:
            str_states = list(map(get_str_sp, (*bra, *ket)))
            header = " 0 5 "+" ".join(str_states)+" {} {}".format(J_range[0], 
                                                                  J_range[-1])
            block = [header+"\n", ]
            for J in J_range:
                vals = ["{:15.10f}".format(v) for v in J_block[J]]
                block.append("    "+" ".join(vals)+"\n")
            matrix_elements_J[(bra, ket)] = block
            order_meJ.append((bra, ket))

with open("onlyDD_Jme_taurus.2b", 'w+') as f:
    f.write("Recoupling J scheme from [{}] file, generated by Python script.\n"
            .format(filename))
    for me_sh in order_meJ:
        f.write("".join(matrix_elements_J[me_sh]))
print("Program finished properly. -------------------------------------------")

#%% Check with the matrix elements from the spherical approximation

file_bench = "test_output_DDonlyD1S_bench.2b"
file_bench = "test_output_DDonlyD1S_bench16O_MZ2.2b" #sp sd
# file_bench = "test_output_DDonlyD1S_bench16O_MZ3.2b" #sp sd pf

me_bench = {}
# me_bench_missing = {}
me_test_missing  = {}
me_test_present  = {}

with open(file_bench, 'r') as f:
    data = f.readlines()[1:]
    
    for line in data:
        if line.startswith(' 0 5 '):
            head_line = line
            header = line.strip().split(" ")
            a, b = int(header[2]), int(header[3])
            c, d = int(header[4]), int(header[5])
            jmin, jmax = int(header[-2]), int(header[-1])
            
            ab1, ab2 = (a, b), (b, a)
            cd1, cd2 = (c, d), (d, c)
            ab_exch  = (sh_2j[a] + sh_2j[b])//2 + 1
            cd_exch  = (sh_2j[c] + sh_2j[d])//2 + 1
            
            comb = [(ab1,cd1, 0, 0),       (ab2,cd1, ab_exch, 0),
                    (ab1,cd2, 0, cd_exch), (ab2,cd2, ab_exch, cd_exch),
                    (cd1,ab1, 0, 0),       (cd1,ab2, 0, ab_exch),
                    (cd2,ab1, cd_exch, 0), (cd2,ab2, cd_exch, ab_exch)]
            
            found = False
            for bra, ket, exch_bra, exch_ket in comb:
                
                if (bra, ket) in matrix_elements_J:
                    found = True
                phs = exch_bra + exch_bra
                phase = [(j, (-1)**(phs + j*((exch_bra!=0) + (exch_ket!=0))))
                            for j in range(jmin, jmax+1)]
                phase = dict(phase)
                if found:
                    me_test_present[(bra, ket)] = []
                    J = jmin
                    break
            
            if not found:
                me_test_missing[((a,b), (c,d))] = [head_line, ]
        
        else:
            
            if found:
                
                mes = (float(x)*phase[J] for x in line.strip().split('\t'))
                
                line = ' '.join(["{:13.10f}".format(x) for x in mes])
                me_test_present[(bra, ket)].append(line+'\n')
                
                J += 1
            else:
                me_test_missing[((a,b), (c,d))].append(line)
            

in_bench = set(list(me_test_present.keys())).union(set(list(me_test_missing.keys())))
me_bench_missing_set = set(list(matrix_elements_J.keys())).difference(in_bench)
print("ME missing in bench (due triaxial deformations)")
order_meJ = sorted(me_bench_missing_set)
print("\n".join([str(x) for x in order_meJ]))

me_bench_missing = {}
for sp_me in me_bench_missing_set:
    me_bench_missing[sp_me] = matrix_elements_J[sp_me]


with open("onlyDD_Jme_taurus_NotInBench.2b", 'w+') as f:
    f.write("Recoupling J scheme from [{}] file, m.e. not in Spherical Bench file[{}].\n"
            .format(filename, file_bench))
    for me_sh in order_meJ:
        f.write("".join(me_bench_missing[me_sh]))


print("End comparison test.       -------------------------------------------")

