"""
Created on Fri Mar  4 19:28:46 2022

@author: Miguel
"""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import os
import shutil
import subprocess
import time

from exe_isotopeChain_axial import DataAxial
from exe_isotopeChain_taurus import DataTaurus
from exe_q20pes_taurus import TemplateDDInp, template_DD_input, zipBUresults
from exe_q20pes_taurus import _setDDTermInput, _printInput4Calculation  # _exportResult (overwriten)
import math as mth
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

class Template:
    interaction    = 'interaction'
    com            = 'com'
    read_red_hamil = 'red_hamil' 
    z    = 'z'
    n    = 'n'
    seed = 'seed'
    b20  = 'b20'
    b22  = 'b22'
    P_T00_J10  = 'P_T00_J10'
    P_T00_J1m1 = 'P_T00_J1m1'
    P_T00_J1p1 = 'P_T00_J1p1'
    P_T10_J00  = 'P_T10_J00'
    P_T1m1_J00 = 'P_T1m1_J00'
    P_T1p1_J00 = 'P_T1p1_J00'
    
    grad_type  = 'grad_type'
    grad_tol   = 'grad_tol'
    eta_grad   = 'eta_grad' 
    mu_grad    = 'mu_grad'
    iterations = 'iters'
    
    
    

PAIRC_EXTENSION = {
    Template.P_T00_J10 : Template.P_T00_J10 .replace("_",""), 
    Template.P_T10_J00 : Template.P_T10_J00 .replace("_",""),
    Template.P_T1p1_J00: Template.P_T1p1_J00.replace("_",""),
    Template.P_T1m1_J00: Template.P_T1m1_J00.replace("_","")}

template = """Interaction   
-----------
Master name hamil. files      {interaction}
Center-of-mass correction     {com}
Read reduced hamiltonian      {red_hamil}
No. of MPI proc per H team    0

Particle Number
---------------
Number of active protons      {z}.00
Number of active neutrons     {n}.00  
No. of gauge angles protons   1
No. of gauge angles neutrons  1

Wave Function   
-------------
Type of seed wave function    {seed} 
Number of QP to block         0
No symmetry simplifications   0
Seed random number generation 0
Read/write wf file as text    0
Cutoff occupied s.-p. states  0.00E-00
Include all empty sp states   0
Spatial one-body density      1
Discretization for x/r        100 0.075
Discretization for y/theta    0   0.00 
Discretization for z/phi      0   0.00 

Iterative Procedure
-------------------
Maximum no. of iterations     {iters} 
Step intermediate wf writing  1
More intermediate printing    0
Type of gradient              {grad_type}
Parameter eta for gradient    {eta_grad:4.3f}E-00
Parameter mu  for gradient    {mu_grad:4.3f}E-00
Tolerance for gradient        {grad_tol:4.3f}E-00

Constraints             
-----------
Force constraint N/Z          1
Constraint beta_lm            1
Pair coupling scheme          1
Tolerance for constraints     1.000E-08
Constraint multipole Q10      1   0.000
Constraint multipole Q11      1   0.000
Constraint multipole Q20      {b20}
Constraint multipole Q21      1   0.000
Constraint multipole Q22      {b22}
Constraint multipole Q30      0   0.000
Constraint multipole Q31      1   0.000
Constraint multipole Q32      0   0.000
Constraint multipole Q33      0   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      1   0.000
Constraint multipole Q42      0   0.000
Constraint multipole Q43      0   0.000
Constraint multipole Q44      0   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     {P_T00_J10}
Constraint pair P_T00_J1m1    {P_T00_J1m1}
Constraint pair P_T00_J1p1    {P_T00_J1p1}
Constraint pair P_T10_J00     {P_T10_J00}
Constraint pair P_T1m1_J00    {P_T1m1_J00}
Constraint pair P_T1p1_J00    {P_T1p1_J00}
Constraint field Delta        0   0.000
"""


#%% main

def _executeProgram(params, output_filename, bp_index, 
                    print_result=True, save_final_wf=True, force_converg=False):
    z, n = params[Template.z], params[Template.n]
    res = None
    if not Template.grad_tol in params:
        params[Template.grad_tol] = 0.005
    try:
        status_fin = ''
        
        ## ----- execution ----
        text = template.format(**params)
        
        with open(DataTaurus.INPUT_FILENAME, 'w+') as f:
            f.write(text)
        str_q20 = '_'.join([str(round(x,3)) for x in bp_index]).replace('.','')
        
        _e = subprocess.call('./taurus_vap.exe < {} > {}'
                                  .format(DataTaurus.INPUT_FILENAME, 
                                          output_filename), 
                              shell=True,
                              timeout=43200) # 1/2 day timeout
        res = DataTaurus(z, n, output_filename)        
        if res == None: return
        # move shit to the folder
        folder_dest = os.getcwd()+'/'+DataTaurus.BU_folder+'/'
        _e = subprocess.call('mv {} {}'.format(output_filename, 
                              folder_dest+output_filename+ f'_Z{z}N{n}_{str_q20}.txt'),
                              shell=True, timeout=8640) # 2.4h timeout
        _e = subprocess.call('cp final_wf.bin '+folder_dest+
                             f'seed_q{str_q20}_Z{z}N{n}.bin',  shell=True)
        if os.path.exists('spatial_density_R.dat'):
            ord_ = f"cp spatial_density_R.dat {folder_dest}densR_z{z}n{n}.dat"
            _e = subprocess.call(ord_, shell=True)
        _e = subprocess.call('rm *.dat', shell=True) # *.red
        
        # refresh the initial function to the new deformation
        if save_final_wf and (res.properly_finished or (not force_converg)):
            _e = subprocess.call('rm initial_wf.bin', shell=True)
            _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
            print("      *** exec. [OK] copied the final wf to the initial wf!")
                
        status_fin = 'X' if not res.properly_finished else '.'
        if print_result:  
            print(" {:2} {:2}  ( {})    {:9.4f}  {:9.4f}  {:7.4f}  {:6.4f},{:6.2f} [{}/{}]"
                  .format(z, n, status_fin, res.E_HFB, res.kin, res.pair, 
                          res.b20_isoscalar, res.gamma,
                          res.iter_max, params[Template.iterations]))
            #print(" >>", datetime.now().time())
    
    except Exception as e:
        print("\n 1>> EXCEP (exe_q20pes_taurus._executeProgram) >>>>>>>  ")
        print(" 1>> current b20 =", str_q20)
        print(" 1>> OUTPUT FILE from TAURUS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                text = f.read()
                print(text)
        else:
            print(" 1 ::: [WARNING] Cannot open output file !!")
        print(" 1<< OUTPUT FILE from TAURUS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(" 1> [",e.__class__.__name__,"]:", e, "<1")
        if res and res.E_HFB == None and not res.properly_finished:
            print(" 1> the result is NULL (final_wf wasn't copied to initial_wf)")
        else:
            print(" 1> result not obtained, return None.")
        #print(res)
        print(" 1<< EXCEP (exe_q20pes_taurus._executeProgram) <<<<<<<  ")
        return None
    
    return res


def _exportResult(results, path_, key_order=None):
    # TODO: modify to store the index of the value
    # key_order = [(0,0), (0,1), ...]
    data = []
    if isinstance(results, list):
        for res in results:
            if res:
                line = res.getAttributesDictLike
                data.append(line+'\n')
    elif isinstance(results, dict):
        ## else is a dictionary and require a key_order
        data = []
        for key_ij in key_order:
            res = results.get(key_ij) #[]
            if res:
                line = res.getAttributesDictLike
                data.append(str(key_ij)+" = "+line+'\n')
    else:
        raise Exception("results must be either list(no key order), or (dict and require key_order list)")
    
    with open(path_, 'w+') as f:
        f.writelines(data)


def _set_deform_for_TES(res_0, b_lims, pair_args, keep_b20_limits=True):
    """ 
        Deform plane and PES starting form the energy minimum.
    ARGS:
      res_0    : <DataTaurus> result for the minimum
      b_lims   : (b20_minimum, b20_maximum, N b20 points (it will be odd number))
      pair_args: (Pair_constrained attribute name, p min, p max, N p points) 
    
      keep_b20_limits=True : <bool>, in case the b20 minima is outside the 
                b max/min range, the deformations will be extended from the new
                b20_minima +/- 1/2 * b20_range in the extended part.
                The argument makes the code to extend the other limit up to the 
                selected in @b_lims (modifying the Nb) or shift to 
                b20 minima -/+ 1/2 * b20_range.
                
    RETURN:
      betas, pairs <dict>:{index_ : deformation}
      deform_array <list>: [ tuple: (i_b, i_p, b[i_b], p[i_p], cpwf, wr_temp_wf])
        
        where: 
      cpwf is the wf to be used for the current calculation
        = 0 (base wf), = 1 (temporary wf for p=p base), = 2 (prev intermediate wf)
      wr_temp_wf comand the temporary wf to be overwriten after the current 
        calculation (when p = p_base and we evaluate a new b deform) = 0, 1 
    """
    
    b1, b2, Nb = b_lims
    pair_constr, p1, p2, Np = pair_args
    assert b1 <= b2, f"B_min={b1:7.5} greater than B_max={b2:7.5f}. STOP!"
    assert p1 <= p2, f"Pair_min={p1:7.5} greater than Pair_max={p2:7.5f}. STOP!"
    Nb = Nb if Nb % 2 == 1 else Nb + 1   ## make an odd number of beta vals
    # Np = Np if Np % 2 == 1 else Np + 1
    b_range = b2 - b1
    p_range = p2 - p1
    db = b_range / Nb
    dp = p_range / Np
    
    # TODO: fix the minimum b_base to the res_0 free
    b_base  = round(res_0.b20_isoscalar, 3)
    p_base  = round(getattr(res_0, pair_args[0], 0.0), 3)
        
    betas = []
    pairs = []
    ## 1.1 set the limits for BETA and shift them to the minima value
    if b_base < b1: # left outter
        b1 = b_base - min(abs(b_base - b1), b_range / 2) # do not over extend the new range
        if keep_b20_limits:
            b_range = b2 - b1
            Nb = int(b_range / db)
            Nb = Nb if Nb % 2 == 1 else Nb + 1   ## make an odd number of beta vals
        else:
            b2 = b_base + (b_range / 2)
    
    elif b_base > b2 : # right outter
        b2 = b_base + min(abs(b_base - b2), b_range / 2)
        if keep_b20_limits:
            b_range = b2 - b1
            Nb = int(b_range / db)
            Nb = Nb if Nb % 2 == 1 else Nb + 1   ## make an odd number of beta vals
        else:
            b1 = b_base - (b_range / 2)
    
    # else and previus cases, shift the beta values to the nearest one
    betas = np.linspace(b1, b2, num=Nb, endpoint=True) #original points
    dq2_prev = b2 - b1
    for i in range(Nb):
        dq2_0 = betas[i] - b_base
        if abs(dq2_0) < abs(dq2_prev):
            dq2_prev = dq2_0
        else: 
            # neares value was the previous
            ii = i - 1
            b_base = betas[ii]
            index_b_lims = [-ii, len(betas) - ii -1]
            #shift the betas
            betas = [(index_b_lims[0] + j, betas[j]-dq2_prev) for j in range(Nb)]
            break
    
    
    ## 1.2 Select PAIR values (does not shift, include 0.0) and  increase the 
    ## range for the pair proportionally if land outside
    if p_base < 0.0:
        raise Exception(" Anomalous Case for First convergence on constraint" 
                        f"{pair_constr} has NEGATIVE value [{p_base:5.3f}] !")
    else:
        if p_base > p2: ## upper out
            p2 = p_base + (p_base - p2)
            # omit the lower limit, extend the Np
            p_range = p2 - p1
            Np = int(p_range / dp)
    
    # else and previus cases, shift the beta values to the nearest one
    pairs = np.linspace(p1, p2, num=Np, endpoint=True) #original points
    dq2_prev = p2 - p1
    for i in range(Np):
        dq2_0 = pairs[i] - p_base
        if abs(dq2_0) < abs(dq2_prev):
            dq2_prev = dq2_0
        else: 
            # neares value was the previous
            ii = i - 1
            p_base = pairs[ii]
            index_p_lims = [-ii, len(pairs) - ii -1]
            # DONT shift the pair values
            pairs = [(index_p_lims[0] + j, pairs[j]) for j in range(Np)]
            break
    
    betas = dict(betas)
    pairs = dict(pairs)
    
    ## 2.1 sort the values from: base -> prolate -> oblate and p_base -> p2 -> p1=0.0
    ## TODO:: 
    deform_array = []
    ## "prolate" part of b20 deformations
    for i_b in range(index_b_lims[1]):
        # positive pair part
        cpwf = 0 if i_b==0 else 1
        wrwf = 1
        for i_p in range(index_p_lims[1]+1):
            deform_array.append((i_b, i_p, betas[i_b], pairs[i_p], cpwf, wrwf))
            cpwf = 2
            wrwf = 0
        
        cpwf = 1
        # negative pair part
        for i_p in range(-1, index_p_lims[0]-1, -1):
            deform_array.append((i_b, i_p, betas[i_b], pairs[i_p], cpwf, wrwf))
            cpwf = 2
        wrwf = 1
    ## "oblate" part of b20 deformations
    for i_b in range(-1, index_b_lims[0]-1, -1):
        # positive pair part
        cpwf = 0 if i_b==-1 else 1
        wrtempwf = 1
        for i_p in range(index_p_lims[1]+1):
            deform_array.append((i_b, i_p, betas[i_b], pairs[i_p], cpwf, wrwf))
            cpwf = 2
            wrwf = 0
        cpwf = 1
        # negative pair part
        for i_p in range(-1, index_p_lims[0]-1, -1):
            deform_array.append((i_b, i_p, betas[i_b], pairs[i_p], cpwf, wrwf))
            cpwf = 2      
        wrwf = 1
    
    print('\n  ****  DEFORMATIONS 2 EVAL.  ********************** ')
    print('ib ip   b_defor p_defor   cp   wrwf  ')
    for ib, ip, b_def, p_def, cpwf, wrwf in deform_array:
        if cpwf == 1:
            print()
        elif cpwf == 0:
            print("   GLOBAL MINIMA COPY ************************ ")
        print(f" {ib:2}, {ip:2} {b_def:+5.3f} {p_def:+5.3f}  -> {cpwf} {wrwf}")
    print('  *******************************  THATS ALL     *** ')
    
    return betas, pairs, deform_array

# ##  TODO: REMOVE  test _set_deform_for_TES
# blims      = (-0.12, 0.12, 10)
# PAIR_CONSTR = Template.P_T10_J00
# pargs   = (PAIR_CONSTR, 0.0, 3.5, 10)
# #
# res_0       = DataTaurus(12, 12, '', empty_data=True)
# res_0.b20_isoscalar = 0.48
# setattr(res_0, PAIR_CONSTR, 0.0)
# #
# betas, pairs, defor_array = _set_deform_for_TES(res_0, blims, pargs, keep_b20_limits=True)
# #
# for ib, ip, b_def, p_def, cpwf, wrwf in defor_array:
#     if cpwf == 1:
#         print()
#     elif cpwf == 0:
#         print("\n GLOBAL MINIMA COPY ************************ ")
#     print(ib, ip, b_def, p_def, cpwf, wrwf)
# print('\n  ******************************* ENDED PROPERLY *** ')
# raise BaseException('')


def _set_constrains(b_base, g_base, pkey, pval):
    constr = {Template.b20  : "0  {:+5.4f}".format(0.0000),
              Template.b22  : "1   {:5.4f}".format(g_base)} ## ignore to get diferent gamma
    
    if b_base != None:
        constr[Template.b20] =  "1  {:+5.4f}".format(b_base)
    
    for pk in Template.__dict__.keys():
        if not pk.startswith('P_T'): continue
        if pk == pkey:
            constr[pkey] = "1  {:+5.4f}".format(pval)
        else:
            constr[pk]   = "0   0.000"
    
    return constr


def _set_pair_constrains_bfixed(b_base, g_base, pair_args, pair_constr):
    """ 
    modification of the previous, b is fixed and all the pair arguments must be 
    given. If b_base is None, pair args are ignored
    """
    constr = {Template.b20  : "0   0.000",
              Template.b22  : "0   0.000"}
    pair_keys = filter(lambda k: k.startswith('P_T'), Template.__dict__.keys())
    
    if b_base != None:
        ## the base seed was fixed, set its deformation
        if g_base == None:
            print("[WARNING] b22 must be given if b20 is given, setted ")
        constr[Template.b20] =  "1  {:+5.4f}".format(b_base)
        constr[Template.b22] =  "1  {:+5.4f}".format(g_base)
    else:
        ## default exportation, all unsetted to get a minimum b2-p deform
        for pk in pair_keys:
            constr[pk] = '0   0.000'
        return constr
    
    for pk in pair_keys:
        if pk not in pair_args: 
            print("[WARNING] pair_key", pk," not in pair_args!, setted zero.")
        pval = pair_args.get(pk, 0.0)
        if pk == pair_constr:
            constr[pk] = "1   {:5.4f}".format(pval)
        else: # condition to unset the constraints of 
            constr[pk] = "0   {:5.4f}".format(pval)
        
    return constr


def _set_pair_for_PES(res_0, pair_constr, p_min, p_max, N_max):
    """ Complete the pair deformations to the left(oblate) and right(prol)
    for the pairing minimum. dq defined by N_max, final total length= N_max+1
    """
    # p_min = max(p_min, 0.0)    
    deform_oblate, deform_prolate = [], []
    dq = round((p_max - p_min) / N_max,  3)
    q0 = getattr(res_0, pair_constr, None)
    if   q0 < p_min: # p0 outside to the left (add the difference to the left)
        p_min = q0 + (q0 - p_min)
    elif q0 > p_max: # p0 outside to the right (add the diff. to the right)
        p_max = q0 + (q0 - p_max)
    
    if  q0 == None: 
        print("[WARNING] _set_pair could not get pait_constraint", pair_constr,
              "  setting to 0.00")
        q0 = 0.00
    deform_oblate.append(q0)
    q = q0
    while (q >  p_min):
        q = q - dq
        if (q > p_min):
            deform_oblate.append(q)
        else:
            deform_oblate.append(p_min)
    q = q0
    while (q < p_max):
        q = q + dq
        if  (q < p_max):
            deform_prolate.append(q)
        else:
            deform_prolate.append(p_max)
    
    return deform_oblate, deform_prolate

def copy_hamiltonian_byZN(z, n, MZ, folder_hamilByZN):
    """  """
    if folder_hamilByZN and os.path.exists(folder_hamilByZN):
        _hamil_name = f"D1S_t0_z{z}n{n}_MZ{MZ}"
        print("  * Searching Hamil sho:",_hamil_name, "in", folder_hamilByZN)
        if _hamil_name+'.sho' in os.listdir(folder_hamilByZN):
            for tail_ in ('.sho', '.2b', '.com'):
                fif_ = folder_hamilByZN+"/"+_hamil_name+tail_
                if os.path.exists(fif_):
                    shutil.copy(fif_, _hamil_name+tail_)
            return _hamil_name, False
    
    print(" [Error] Hamil sho:",_hamil_name, "not found in ", folder_hamilByZN, " CONTINUE.")
    return '', True
        

def mainGenerateSeeds(nucleus, interaction, 
                      ZNcore=(0,0), seed_=0, dd=False, dest_folder='seeds',
                      folder_hamilByZN=None):
    """
    :nucleus: <list> global Z,N for the nucleus (including the core)
    :interaction: name of the hamiltonian file
        Optionals
    :ZNcore:  <tuple> if core is present, z0,n0 will be substracted from Z,N
    :seed_:  is the default seed to begin, it compromises the self-consistent 
        symmetries
    :dd: sets the density dependent term or not
    :dest_folder: place for the final_wf to be placed
    
    default convention for final w.f.: final_z[]n[].bin
    """
    output_filename = 'aux_output'
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    DataTaurus.BU_folder = dest_folder
    DataTaurus.setUpFolderBackUp()
    
    # create a spherical/unbounded seed to proceed
    _iter_base = 800
    
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.seed : seed_,
        Template.grad_type: 1,
        Template.grad_tol : 0.001,
        Template.eta_grad : 0.03,#0.1,
        Template.mu_grad  : 0.2,# 0.1,
        Template.iterations : _iter_base
    }
    RDIM_0, OMEG_0 = 12, 14
    
    for k_constr, val in _set_pair_constrains_bfixed(None, None, None, None).items():
        kwargs[k_constr] = val
    _setDDTermInput(dd, RDIM_0, OMEG_0)
    
    print(HEAD)
    
    for z, n in nucleus:
        ## Appendix to evaluate minimum solution for a certain hamiltonian by ZN
        _hamil_name, skip_ = copy_hamiltonian_byZN(z, n, folder_hamilByZN)
        if skip_: 
            continue
        kwargs[Template.interaction] = _hamil_name
        
        print(f" * Convergence (seed {kwargs[Template.seed]}), interact={interaction}")  ## ***************** 
        z -= ZNcore[0]
        n -= ZNcore[1]
        kwargs[Template.z] = z
        kwargs[Template.n] = n
        kwargs[Template.iterations] = _iter_base
        
        copyFWF = True
        for iter_ in range(1, 4):
            res_0 = _executeProgram(kwargs, output_filename, (0.0, 0.0), 
                                    save_final_wf=True)
            if (res_0 != None):
                if res_0.properly_finished:
                    break
                else:
                    kwargs[Template.seed] = 1 # take the last step (assumme it went ok)
            else:
                if iter_ == 3:
                    print("[ERROR], after 4 tries the calculation STOP for", z, n)
                    copyFWF = False
                else:
                    print(f"({z}, {n}): {res_0.E_HFB:6.2f},")

            # kwargs[Template.eta_grad]   -= 0.007 * iter_
            # kwargs[Template.eta_grad]    = max(kwargs[Template.eta_grad], 0.001)
            kwargs[Template.iterations] += 150 * (iter_ + 1)
            print(" [WARNING] 1st step non converged, next try:", 
                      iter_, kwargs[Template.eta_grad])
        if copyFWF:
            ordr = 'cp final_wf.bin {}/final_z{}n{}.bin'.format(dest_folder,z,n)
            _e = subprocess.call(ordr, shell=True)
        else:
            print("  not copied.")
        
    print("   ... done:")  ## ************************************************
    for f in filter(lambda x: x.endswith('.bin'), os.listdir(dest_folder)):    
        print(" ", f)
    
    

def mainLinuxFixedStep_Bfixed(z, n, interaction, pair_constr, 
                              p_min=0.0, p_max=5.0, N_max=50, seed_=0, dd=True):
    """ 
        Old process that sets an even single-evaluated step over b range
    voidDD_path is the equivalent of the DataTaurus.export_list_results for the 
    output of the final calculation
    """
    #%% Executing the process, run the list of isotopes
    #
    
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    
    b20, b22 = 0.000, 0.000
    results = []
    
    if seed_ == 1:
        # order_ = 'cp seedsKB3G/final_z{}n{}.bin initial_wf.bin'.format(z,n)
        order_ = 'cp seedsPFx1/final_z{}n{}.bin initial_wf.bin'.format(z,n)
        _e = subprocess.call(order_, shell=True)
        print("\n EEEEEH! REUSING THE WAVEFUNCTION ****** \n\n"*3)
    print(HEAD)
    # create a spherical/unbounded seed to proceed
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.z : z,     Template.n : n,
        Template.seed : seed_,
        Template.grad_type : 1,
        Template.grad_tol  : 0.001,
        Template.eta_grad : 0.03,#0.1,
        Template.mu_grad  : 0.2,# 0.1,
        Template.iterations : 500 if seed_ != 1 else 0,
    }
    
    for k_constr, val in _set_pair_constrains_bfixed(None, None, None, None).items():
        kwargs[k_constr] = val
    RDIM_0, OMEG_0 = 10, 12
    _setDDTermInput(dd, RDIM_0, OMEG_0)
    
    print(" * first convergence (seed{}), interact={}" 
            .format(kwargs[Template.seed], interaction))  ## ***************** 
    for iter_ in range(1, 4):
        res_0 = _executeProgram(kwargs, output_filename, (0.0, 0.0), 
                                save_final_wf=True)
        if (res_0 != None) or res_0.properly_finished:
            break
        else:
            if iter_ == 3:
                print("[ERROR], after 4 tries the calculation STOP for", z, n)
                return
        kwargs[Template.eta_grad]   -= 0.007 * iter_
        kwargs[Template.eta_grad]    = max(kwargs[Template.eta_grad], 0.001)
        kwargs[Template.iterations] += 150 * (iter_ + 1)
        print(" [WARNING] 1st step non converged, next try:", iter_, kwargs[Template.eta_grad])
    _e = subprocess.call('cp final_wf.bin initial_base.bin', shell=True)
    print("   ... done.")  ## ************************************************

    b20  = res_0.b20_isoscalar
    b22  = res_0.b22_isoscalar
    pair_args = {}
    pair_keys = filter(lambda k: k.startswith('P_T'), Template.__dict__.keys())
    pair_args = dict([(pk, getattr(res_0, pk, 0.0)) for pk in pair_keys])
    # ###
    deform_oblate, deform_prolate  = _set_pair_for_PES(res_0, pair_constr, 
                                                       p_min, p_max, N_max)
    #update kwargs
    kwargs[Template.read_red_hamil] = 1
    kwargs[Template.seed] = 1
    kwargs[Template.grad_type]  = 1
    kwargs[Template.grad_tol]   = 0.005
    _default_grad = {Template.eta_grad: 0.01,
                     Template.mu_grad:  0.20, 
                     Template.iterations: 600}
    for k_, val in _set_pair_constrains_bfixed(None, None, pair_args, pair_constr).items():
        kwargs[k_] = val
    ## WARNING! compromising start point
    try:
        _printInput4Calculation(dp=deform_prolate[1]-deform_prolate[0], N=N_max,
                                b_LIMS=(deform_oblate[-1],deform_prolate[-1]), 
                                rdim=RDIM_0, omega=OMEG_0, deform_Obla=deform_oblate,
                                deform_prol=deform_prolate,**kwargs)
    except BaseException as e:
        print("[error_printing] 4calculation: trace:\n", e.__class__, e)
            
    _setDDTermInput(dd, RDIM_0, OMEG_0)
    for i_deform, deform in enumerate((deform_prolate, deform_oblate)):             
        # copy it.
        _e = subprocess.call('cp initial_base.bin initial_wf.bin', shell=True)
        
        for pc_val in deform:
            ## modify value of the pairing constraint.
            kwargs[pair_constr] = "1   {:5.4f}".format(pc_val)
            # reset gradient options            
            kwargs[Template.iterations] = _default_grad[Template.iterations]
            kwargs[Template.eta_grad]   = _default_grad[Template.eta_grad]
            kwargs[Template.mu_grad]    = _default_grad[Template. mu_grad]
            for iter_ in range(1, 4): # repeat if not converges
                res = _executeProgram(kwargs, output_filename, (b20 ,pc_val),
                                      save_final_wf=True) 
                # note: this could affect if the result is none or bad converged
                if (res != None) and res.properly_finished:
                    # _e = subprocess.call()
                    break
                else:
                    if iter_ == 3:
                        print("[ERROR], after 3 tries the calculation STOP for", z, n)
                kwargs[Template.mu_grad]   = 0.01 # or 0 to avoid inercia effect
                kwargs[Template.eta_grad] -= 0.002 * iter_
                kwargs[Template.eta_grad]  = max(kwargs[Template.eta_grad], 0.005)
                kwargs[Template.iterations] += 50 * (iter_ + 1)
                print(" [WARNING] 1st step non converged, next try:", iter_, kwargs[Template.eta_grad])
                
            # res = _executeProgram(kwargs, output_filename, (b20 ,pc_val),
            #                       save_final_wf=True)
            print(f"res.{pair_constr} = {getattr(res, pair_constr, None)}")
            if res == None:
                continue # dont save empty result
            if i_deform == 1: #oblate case (grow to the left)
                results.insert(0, res) #grow in order [-.5, -.4, ..., .0,..., +.4, +.5] 
            else:   # prolate case (append 2the right)
                results.append(res)   
                        
            # intermediate print
            _exportResult(results, DataTaurus.export_list_results)
    
    # ## ------ end exec.  -----
    _exportResult(results, DataTaurus.export_list_results)
    print("   ** generate File in:", DataTaurus.export_list_results)
    
def mainLinuxFixedStep_BackNForth(z, n, interaction, pair_constr, 
                              p_min=0.0, p_max=5.0, N_max=50, seed_=0, dd=True):
    """ 
        The range and deformations are the same as in _BstepFixed. 
    In this case we advance until the p-limits and then go back to the minima,
    getting the lower energy.
    """
    #%% Executing the process, run the list of isotopes
    #
    
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    
    b20, b22 = 0.000, 0.000
    results = []
    
    if seed_ == 1:
        _FOL_SEEDS = 'hamil_folder'# 'seedsMZ4'# 'seedsKB3G'#
        order_ = 'cp {}/final_z{}n{}.bin initial_wf.bin'.format(_FOL_SEEDS,z,n)
        _e = subprocess.call(order_, shell=True)
        print("\n EEEEEH! REUSING THE WAVEFUNCTION ****** \n\n"*3)
        print(order_)
    print(HEAD)
    # create a spherical/unbounded seed to proceed
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.z : z,     Template.n : n,
        Template.seed : seed_,
        Template.grad_type : 1,
        Template.grad_tol  : 0.001,
        Template.eta_grad : 0.03,#0.1,
        Template.mu_grad  : 0.2,# 0.1,
        Template.iterations : 500 if seed_ != 1 else 500, ## TODO: CHANGE to 0
    }
    
    for k_constr, val in _set_pair_constrains_bfixed(None, None, None, None).items():
        kwargs[k_constr] = val
    RDIM_0, OMEG_0 = 12, 14
    _setDDTermInput(dd, RDIM_0, OMEG_0)
    
    print(" * first convergence (seed{}), interact={}" 
            .format(kwargs[Template.seed], interaction))  ## ***************** 
    for iter_ in range(1, 4):
        res_0 = _executeProgram(kwargs, output_filename, (0.0, 0.0), 
                                save_final_wf=True)
        if (res_0 != None) or res_0.properly_finished:
            break
        else:
            if iter_ == 3:
                print("[ERROR], after 4 tries the calculation STOP for", z, n)
                return
        kwargs[Template.eta_grad]   -= 0.007 * iter_
        kwargs[Template.eta_grad]    = max(kwargs[Template.eta_grad], 0.001)
        kwargs[Template.iterations] += 150 * (iter_ + 1)
        print(" [WARNING] 1st step non converged, next try:", iter_, kwargs[Template.eta_grad])
    _e = subprocess.call('cp final_wf.bin initial_base.bin', shell=True)
    print("   ... done.")  ## ************************************************
    
    b20  = res_0.b20_isoscalar
    b22  = res_0.b22_isoscalar
    pair_args = {}
    pair_keys = filter(lambda k: k.startswith('P_T'), Template.__dict__.keys())
    pair_args = dict([(pk, getattr(res_0, pk, 0.0)) for pk in pair_keys])
    # ###
    deform_oblate, deform_prolate  = _set_pair_for_PES(res_0, pair_constr, 
                                                       p_min, p_max, N_max)
    ## step to order forward-backward the deformations, adding the index and 
    ## back-traking state (=1 if back)
    def_obl, def_pro = [], []
    for i in range(len(deform_oblate)):
        i2 = len(def_obl) //2
        def_obl.insert(i2, (i,1,deform_oblate[i]))
        def_obl.insert(i2, (i,0,deform_oblate[i]))
        
    for i in range(len(deform_prolate)):
        def_pro.insert(len(def_pro) - i, (i,1,deform_prolate[i]))
        def_pro.insert(i, (i,0,deform_prolate[i]))
    
    #update kwargs
    kwargs[Template.read_red_hamil] = 1
    kwargs[Template.seed] = 1
    kwargs[Template.grad_type]  = 1
    kwargs[Template.grad_tol]   = 0.005
    _default_grad = {Template.eta_grad: 0.01,
                     Template.mu_grad:  0.20, 
                     Template.iterations: 600}
    for k_, val in _set_pair_constrains_bfixed(None, None, pair_args, pair_constr).items():
        kwargs[k_] = val
    ## WARNING! compromising start point
    try:
        _printInput4Calculation(dp=deform_prolate[1]-deform_prolate[0], N=N_max,
                                b_LIMS=(deform_oblate[-1],deform_prolate[-1]), 
                                rdim=RDIM_0, omega=OMEG_0, deform_Obla=deform_oblate,
                                deform_prol=deform_prolate,**kwargs)
    except BaseException as e:
        print("[error_printing] 4calculation: trace:\n", e.__class__, e)
    
    _setDDTermInput(dd, RDIM_0, OMEG_0)
    results = [[], []]
    for i_deform, deform in enumerate((def_pro, def_obl)):             
        # copy it.
        _e = subprocess.call('cp initial_base.bin initial_wf.bin', shell=True)
        
        for i, back_track ,pc_val in deform:
            ## modify value of the pairing constraint.
            kwargs[pair_constr] = "1   {:5.4f}".format(pc_val)
            # reset gradient options            
            kwargs[Template.iterations] = _default_grad[Template.iterations]
            kwargs[Template.eta_grad]   = _default_grad[Template.eta_grad]
            kwargs[Template.mu_grad]    = _default_grad[Template. mu_grad]
            for iter_ in range(1, 4): # repeat if not converges
                res = _executeProgram(kwargs, output_filename, (b20 ,pc_val),
                                      save_final_wf=True) 
                # note: this could affect if the result is none or bad converged
                if (res != None) and res.properly_finished:
                    # _e = subprocess.call()
                    break
                else:
                    if iter_ == 3:
                        print("[ERROR], after 3 tries the calculation STOP for", z, n)
                
                kwargs[Template.mu_grad]   = 0.01 # or 0 to avoid inercia effect
                kwargs[Template.eta_grad] -= 0.002 * iter_
                kwargs[Template.eta_grad]  = max(kwargs[Template.eta_grad], 0.005)
                kwargs[Template.iterations] += 50 * (iter_ + 1)
                print(" [WARNING] 1st step non converged, next try:", 
                      iter_, kwargs[Template.eta_grad])
                
            print(f"res.{pair_constr} = {getattr(res, pair_constr, None)}")
            if res == None:
                print(" [Error] resul is None for P_JT=",pc_val," #",i)
                continue # dont save empty result
            
            
            
            if bool(back_track):
                if i_deform == 1:
                    k = len(deform)//2 - 1 - i                    
                    if res.E_HFB < getattr(results[i_deform][k],'E_HFB',1e+30):
                        results[i_deform][k] = res
                else: # prolate
                    if res.E_HFB < getattr(results[i_deform][i],'E_HFB',1e+30):
                        results[i_deform][i] = res
            else:
                if i_deform == 1: #oblate case (grow to the left)
                    results[i_deform].insert(0, res)
                    #grow in order [-.5, -.4, ..., .0,..., +.4, +.5] 
                else:   # prolate case (append 2the right)
                    results[i_deform].append(res)   
    
            # ob = deepcopy()
            final_results = results[1] + results[0]
                        
            # intermediate print
            _exportResult(final_results, DataTaurus.export_list_results)
    
    # ## ------ end exec.  -----
    _exportResult(final_results, DataTaurus.export_list_results)
    print("   ** generate File in:", DataTaurus.export_list_results)


def mainLinuxEvenBetaPair(z, n, interaction, b_lims=(-0.1, 0.1, 6), dd=True, 
                          **kwargs):
    """ 
        Upgrade of mainLinuxEvenBetaPair(), the deformation loops for b, p are
    calculated from a free starting point, the order and exportation are more
    clear than the old one.
    """
    ## 0. SET UP THE VARIABLES ************************************************
    
    output_filename = 'aux_output'
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      beta,  pair"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    
    b_base = 0.000
    g_base = 0.000
    p_base = 0.000
    results = {}
    
    print("kwargs input =", kwargs)
    if not 'seed' in kwargs: 
        print(" WARNING. unspecified SEED, seed will be general: 0")
    elif kwargs['seed'] == 1: 
        print("\n EEEEEH! REUSING THE WAVEFUNCTION ****** \n\n"*3)
    seed_ = kwargs.get('seed', 0)
    pair_args    = ()
    pair_constr  = None
    PAIR_VARS = list((k for k in Template.__dict__.keys() if k.startswith('P_T')))
    
    vals = kwargs['pJT']
    if vals[0] in PAIR_VARS:            
        
        pair_args   = vals #(k, vals[0], vals[1], vals[2])
        pair_constr = vals[0]
        p_base      = min(vals[1:3])
    print(" PAIR_VARS", list(PAIR_VARS))
    print(" pair_constr, pair_args, p_base:")
    print(pair_constr, pair_args, p_base)
    print(" b_lims :: ",b_lims)
    print(" seed_type=", seed_)
    
    print(HEAD)
    ## 1.  CREATE UNBOUNDED SEED TO PROCEED ***********************************
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.z : z,     Template.n : n,
        Template.seed : seed_,
        Template.grad_type : 1,
        Template.grad_tol  : 0.005, #0.005,
        Template.eta_grad  : 0.01,
        Template.mu_grad   : 0.1,
        Template.iterations : 500,
    }
    constr_kwargs = _set_constrains(None, g_base, None, p_base)
    for k,v in constr_kwargs.items():
        kwargs[k] = v
    
    RDIM_0, OMEG_0 = 12, 14
    _setDDTermInput(dd, rdim=RDIM_0, omega=OMEG_0)
    
    print()
    print(" * first convergence (seed{}), interact={}"
            .format(kwargs[Template.seed], interaction))
    for iter_ in range(1, 4):
        res_0 = _executeProgram(kwargs, output_filename, (-1,-1))
        if res_0 and res_0.properly_finished:
            break
        else:
            if iter_ == 3:
                print("[ERROR], after 4 tries the calculation STOP for", z, n)
                return
        kwargs[Template.eta_grad]   -= 0.007 * iter_
        kwargs[Template.eta_grad]    = max(kwargs[Template.eta_grad], 0.001)
        kwargs[Template.iterations] += 150 * (iter_ + 1)
        print(" [WARNING] 1st step non converged, next try:", iter_, kwargs[Template.eta_grad])
    _e = subprocess.call('cp final_wf.bin ini_partial.bin', shell=True)
    _e = subprocess.call('cp final_wf.bin ini_base.bin', shell=True)
    print("   ... done.\n")
      
    ## 2. GET BETAS AND PAIRS SORTED AND THE COPY INDICATIONS *****************
    _args = res_0, b_lims, pair_args
    betas, pairs, deforms  = _set_deform_for_TES(*_args, keep_b20_limits=False)
    
    db = betas[1] - betas[0]
    dp = pairs[1] - pairs[0]
    b_lims = min(betas.keys()), max(betas.keys())
    p_lims = min(pairs.keys()), max(pairs.keys())
    
    ## WARNING! compromising start point
    _printInput4Calculation(dq = round(db,4), dp = round(dp, 4), 
                            b_LIMS=b_lims, p_LIMS=p_lims, 
                            dimensions_calc=(len(betas), len(pairs)), 
                            rdim=RDIM_0, omega=OMEG_0, **kwargs)  
    
    # Fix the execution parameters to start from previous steps
    kwargs[Template.seed] = 1
    kwargs[Template.read_red_hamil] = 1
    kwargs[Template.grad_type]  = 0
    kwargs[Template.grad_tol]   = 0.005
    kwargs[Template.eta_grad]   = 0.01
    kwargs[Template.mu_grad]    = 0.3
    kwargs[Template.iterations] = 700
    key_order = []
    ## 3. COMPUTATION OF ALL DEFORMATIONS *************************************
    print(HEAD)
    for i_b, i_p, b_def, p_def, cpwf, overwr_wf in deforms:
        
        if cpwf   == 0:
            _e = subprocess.call('cp ini_base.bin initial_wf.bin', shell=True)
        elif cpwf == 1:
            _e = subprocess.call('cp ini_partial.bin initial_wf.bin', shell=True)
        elif cpwf == 2:
            pass # previous step copied (or not if error) the final_wf
        
        _setDDTermInput(dd, rdim=RDIM_0, omega=OMEG_0)
        constr_kwargs[Template.b20] = "1   {:5.4f}".format(b_def) 
        constr_kwargs[pair_constr]  = "1   {:5.4f}".format(p_def)
        
        for k,v in constr_kwargs.items(): # rewrite the constrains
            kwargs[k] = v
        
        res = _executeProgram(kwargs, output_filename, (b_def,p_def),
                              save_final_wf=True)
        print("ij:[{},{}/{},{}] res: beta={} pair={}"
                 .format(i_b-b_lims[0], i_p-p_lims[0], len(betas),len(pairs),
                         getattr(res,'b20_isoscalar',-1), 
                         getattr(res, pair_constr, -1)))
        # overwrite temporal p=p base wf
        if overwr_wf:
            e = subprocess.call('cp final_wf.bin ini_partial.bin', shell=True)
            
        if res == None:
            continue # dont save empty result

        ij_key = (i_b, i_p)
        key_order.append(ij_key)
        results[ij_key] = res   
        # intermediate print
        _exportResult(results, DataTaurus.export_list_results, key_order)
        
    
    # ## ------ end exec.  -----
    _exportResult(results, DataTaurus.export_list_results, key_order)
    print("   ** generate File 1st convergence in:", DataTaurus.export_list_results)    


def plot_PairSurfaces(attr2plot, title='', zn=None, export_details=None, **files):
    """ plot of 2D surfaces, several Pairings accepted to print in common, 
    zn=(z, n) to set in title"""
    x_vals = {}
    y_vals = {}
    
    results = {}
    fig , ax = plt.subplots()
    z,n= None, None
    if zn:
        z, n = zn[0], zn[1]
        
    ## TODO: remove for coping the files:   
        
    for pair_constr, file_ in files.items():
        if not os.path.exists(file_):continue
        results[pair_constr] = []
        x_vals [pair_constr] = []
        y_vals [pair_constr] = []
        
        with open(file_, 'r') as f:
            for line in f.readlines():
                res = DataTaurus(None, None, None, True)
                res.setDataFromCSVLine(line)
                results[pair_constr].append(res)
                x_vals[pair_constr].append( getattr(res, pair_constr, -1))
                y_vals[pair_constr].append( getattr(res, attr2plot, -99999))
                        
        
        ax.plot(x_vals[pair_constr], y_vals[pair_constr], '.-', label=pair_constr)
        if not z:
            z,n = res.z, res.n
        else:
            assert z == res.z and n== res.n, "mixed nuclei plots, abort. (Implement me!)"
    
    title = title if title else f"{attr2plot} z,n=({z},{n}) coupling strengths"
    ax.set_xlabel("Pair coup. value")
    ax.set_ylabel(attr2plot)
    ax.set_title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if export_details:
        fld_, interaction = export_details
        plt.savefig(f"{fld_}{attr2plot}_z{z}n{n}_{interaction}.png")
    plt.show()


nuclei = {2:'He', 4:'Be', 6:'C', 8:'O', 10:'Ne', 12:'Mg', 14:'Si', 16:'S', 18:'Ar', 20:'Ca',
          22:'Ti', 24: 'Cr', 26:'Fe', 28: 'Ni',
          34:'Se', 36:'Kr', 38:'Sr', 40: 'Zr'}

nucleus = [
    (12, 12), (12, 10), (12, 8), #(12, 6),  
    (12, 14), (12, 16), (12, 18),
    #(12, 20), (12, 22), (12, 24), (12, 26), (12, 28), (12, 30),
] ## usdb

# nucleus = [
#     ( 8, 8), ( 8,10), ( 8,12),
#     (10,10), (10,12), (10,14), (10,16),
    # (14,12), (14,14),
    # ]


# nucleus = [
#     # (20, 20),  # Ca
#     # (20, 22),
#     # (20, 24),
#     # (20, 26),
#     # (20, 28),
#     # (20, 30),
#     (22, 20), # Ti
#     (22, 22),
#     (22, 24),
#     (22, 26),
#     (22, 28),
#     (22, 30),
#     (24, 20), # Cr
#     (24, 22),
#     (24, 24),
#     (24, 26),
#     (24, 28),
#     (24, 30),
#     (24, 32),
#     (26, 24), # Fe
#     (26, 26),
#     (26, 28),
#     (26, 30),
#     (26, 32),
#     (28, 24), # Ni
#     (28, 26), 
#     (28, 28),
#     (28, 30),
#     (28, 32),
#     (30, 24), # Zn
#     (30, 26), 
#     (30, 28),
#     (30, 30),
#     (30, 32),
# ]  ## kb3g

if __name__ == '__main__':
    
    b_lenght = 1.8
    z = 12 #6 #
    n = 12 #8 #
    
    tail = ''
    tail = 'D1S'
    
    # interaction = 'D1S_t0_SPSDPF' #'D1S_t0_SPSDPF' # 'B1_SPSD' #'0s0p1s0d'  #
    interaction = 'D1S_t0_MZ5'
    # interaction = 'D1S_t0_PF'
    # interaction = 'kb3g.a42'
    # interaction = 'usdb'
    if interaction in ('usdb','kb3g.a42'): tail = interaction[:4]
    
    if not os.getcwd().startswith('C:'):    #
        print()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('%%%   Running PES with Taurus_VAP, hamilonian:', interaction)
        print('%%%     !!! CHECK, CHECK MZ:', interaction.upper())
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print("%%%   GLOBAL START:", datetime.now().time())
        print("%%%")
        
        ## Execute seed collection for pair P_JT 1d plots
        # mainGenerateSeeds(nucleus, interaction, 
        #                   ZNcore=(0,0), seed_=5, dd=True, 
        #                   folder_hamilByZN='hamilsMZ4_Mg')
        # raise Exception("%%% Linux Process seed generation Finished:", datetime.now().time())
        
        for z, n in nucleus:
            print("%%%   Start mainLinux: ", datetime.now().time())
            print("%%%   PES for Z={} N={}".format(z,n))
            
            interaction, skip_ = copy_hamiltonian_byZN(z, n, 5, 'hamil_folder')
            if skip_: 
                continue
                        
            # TODO: Fix Pairing Constraint
            PAIR_CONSTR = Template.P_T10_J00
            PAIR_CONSTR = Template.P_T1p1_J00
            PAIR_CONSTR = Template.P_T1m1_J00
            PAIR_CONSTR = Template.P_T00_J10
            # z -= 20
            # n -= 20
            
            print("%%%   Constraining Pairng J,T: ", PAIR_CONSTR)
            # EXPORT_FILE = "export_PBTESz{}n{}_{}_{}.txt"\
            #                     .format(z,n, tail, PAIR_CONSTR.replace("_",""))
            EXPORT_FILE = "export_PSz{}n{}_{}_{}.txt"\
                                .format(z,n, tail, PAIR_CONSTR.replace("_",""))
            DataTaurus.export_list_results = EXPORT_FILE
            print("%%%")
            
            
            # mainLinuxEvenBetaPair(z, n, interaction, 
            #                       b_lims = (-0.4, 0.4, 9),
            #                       pJT    = (PAIR_CONSTR, 0.0, 3.5, 14),
            #                       seed   = 1,)
            # mainLinuxFixedStep_Bfixed(z, n, interaction, PAIR_CONSTR,
            mainLinuxFixedStep_BackNForth(z, n, interaction, PAIR_CONSTR,
                                          p_min=-0.05, p_max=2.0, N_max=41, # 25 
                                          seed_=1, dd=True)
            
            # export the back up results into a zip folder.
            zipBUresults(z,n,interaction,PAIR_CONSTR)
            print("%%%   End mainLinux: ", datetime.now().time())
    else:
        #%% process in windows
        
        RESCALE_TES = False #True        
        
        PAIR_CONSTR = Template.P_T10_J00
        PAIR_CONSTR = Template.P_T00_J10
        PAIR_CONSTR = Template.P_T1p1_J00
        PAIR_CONSTR = Template.P_T1m1_J00
        z = 10  #6 #
        n = 16  #8 #
        
        # z = 0 #6 #
        # n = 8 #8 #
        
        # z, n = 2, 10
        
        results_taur = []
        index_resuts = {}
        #from exe_isotopeChain_axial import DataAxial
        PESfolder = 'PN_mixingD1STests/MZ4/'
        # PESfolder = 'PN_mixingD1STests/MZ5/'
        # PESfolder = 'PN_mixingD1STests/KB3G/'
        # PESfolder = 'PN_mixingD1STests/USDB/'
        # PESfolder = 'PN_mixingD1STests/MZ4-Isoscalar-Seed0_A28/'
        # PESfolder = 'PN_mixingD1STests/MZ4-Isovector-Seed5/'
        # PESfolder = 'PN_mixingD1STests/Cr54_seed3/'
        PESfolder = 'PN_mixingD1STests/Mg_MZ4/'
        
        # TODO: plot        
        attr_ = 'E_HFB'
        # attr_ = 'hf'
        # attr_ = Template.P_T10_J00
        # attr_ = Template.P_T1p1_J00
        # attr_ = Template.P_T1m1_J00
        # attr_ = Template.P_T00_J10
        # attr_ = Template.P_T00_J1p1
        # attr_ = Template.P_T00_J1m1
        # attr_ = 'kin'
        # attr_ = 'pair'
        # attr_ = 'pair_nn'
        # attr_ = 'pair_pp'
        # attr_ = 'pair_pn'
        # attr_ = 'beta'
        # attr_ = 'b20_isoscalar'
        # attr_ = 'var_p'
        # attr_ = 'gamma'
        # attr_ = 'Jz'
        # attr_ = 'var_n'
        
        # attr_ = 'iter_time_seconds'
        
        if True:
            # for z, n in nucleus:
            z, n = 12, 18
                
            PESfolder = 'PN_mixingD1STests/MZ4_EHFB_vs_PCs_b2Unconstrained/'
            PESfolder = 'PN_mixingD1STests/MZ4_EHFB_vs_PCs/'
            PESfolder = 'PN_mixingD1STests/MZ4_EHFB_vs_PCs/gradient1/' #'_backNForth/'
            PESfolder = 'PN_mixingD1STests/MZ4_EHFB_vs_PCs/gradient1_backNForth/'
            # PESfolder = 'PN_mixingD1STests/D1S_PF_bc40/'
            PESfolder = 'PN_mixingD1STests/D1S_PF_bEqbc/'
            PESfolder = 'PN_mixingD1STests/Mg_MZ4/'
            PESfolder = 'PN_mixingD1STests/Mg_MZ5/'
            # PESfolder = 'PN_mixingD1STests/SDnuclei_MZ4/'
            
            # PESfolder = 'PN_mixingD1STests/MZ3_EHFB_vs_PCs/'
            # PESfolder = 'PN_mixingD1STests/USDB/PS_surf_Mg/'
            # z, n =z - 8, n - 8
            # PESf older = 'PN_mixingD1STests/KB3G/PS_surf_Fe/'
            # z, n = z - 20, n - 20
            # if z != 2: continue
            
            files_ = [
                Template.P_T00_J10, 
                Template.P_T1p1_J00, 
                Template.P_T1m1_J00,
                Template.P_T10_J00
                      ]
            for i, pc in enumerate(files_):
                fn = f"export_PSz{z}n{n}_{tail}_{PAIRC_EXTENSION[pc]}.txt"
                files_[i] = (pc, PESfolder+fn)
            files_ = dict(files_)
            plot_PairSurfaces(attr_, zn=(z,n), 
                                # export_details=(PESfolder, 'D1S'), 
                              **files_)
            raise Exception('STOP')
        
        PESfolder = 'BU_pair_results/'
        IMPORT_FILE = "export_PBTESz{}n{}_{}_{}.txt"\
                            .format(z,n, tail, PAIR_CONSTR.replace("_",""))
        DataTaurus.export_list_results = IMPORT_FILE
        import_file_Tau = PESfolder + DataTaurus.export_list_results        
        
        ## get results from export "csv" 
        IJ = []
        BP = {}
        Bmax, Pmax = 0.0, 0.0
        Bmin, Pmin = 0.0, 0.0
        B_dims = [0,0] # bdims
        P_dims = [0,0] # bdims
        print("IMPORTING: ", import_file_Tau)
        if os.path.exists(import_file_Tau):
            
            with open(import_file_Tau, 'r') as f:
                data = f.readlines()
                for line in data:
                    index_, line = line.split(' = ')
                    index_ = tuple([int(x) for x in index_[1:-1].split(',')])
                    
                    res = DataTaurus(None, None, None, True)
                    res.setDataFromCSVLine(line)
                    # index_ = index_[1], index_[0] ## uncomment if the beta-pair indexes are exchanged
                    results_taur.append((index_, res))
                    
                    # print(index_, ':', res.E_HFB,',')
                    B_dims = [min(index_[0], B_dims[0]), 
                              max(index_[0], B_dims[1])]
                    P_dims = [min(index_[1], P_dims[0]), 
                              max(index_[1], P_dims[1])]
            
            #P_dims[1] -= 3
            results_taur = dict(results_taur)
            
            ## Complete the missing files in order previous p
            real_keys = set(results_taur.keys())
            for i in range(B_dims[0], B_dims[1]+1):
                for j in range(P_dims[1]+1):
                    if not (i,j) in results_taur:
                        results_taur[(i,j)] = results_taur[(i,j-1)]
                for j in range(-1, P_dims[0]-1, -1):
                    if not (i,j) in results_taur:
                        results_taur[(i,j)] = results_taur[(i,j+1)]
            print("Copied keys: ", set(results_taur.keys()).difference(real_keys))
            aux = []
            # TODO: sort the data 
            #
            prev_res  = None
            key_order = []
            
            for i in range(B_dims[0], B_dims[1]+1):
                for j in range(P_dims[0], P_dims[1]+1):
                    key_order.append((i, j))
                    index_resuts[(i,j)] = len(key_order) - 1
            
            b_obl_prol = (B_dims[0], B_dims[1]+1, 1),
            if B_dims[0] < 0:
                if B_dims[1] < 0:
                    b_obl_prol = (B_dims[0], B_dims[1]-1, -1),
                else:
                    b_obl_prol = (-1,B_dims[0]-1,-1), (0,B_dims[1]+1,1)
            
            ## order and from oblate-prolate, p=0 to p=max, starting from b20=0 to prolate
            for bmin, bmax, stp in b_obl_prol:
                for i in range(bmin, bmax, stp):
                    pa_lims = P_dims[0], P_dims[1]+1, 1
                    if  i < 0:
                        pa_lims = P_dims[1], P_dims[0]-1, -1
                    
                    for j in range(*pa_lims):
                        index_ = (i, j)
                        
                        if index_ not in results_taur:
                            if stp == -1:
                                aux.insert(0, (index_, prev_res) )
                            else:
                                aux.append( (index_, prev_res) )
                        else:
                            if stp == -1:
                                aux.insert(0 , (index_, results_taur[index_]))
                            else:
                                aux.append( (index_, results_taur[index_]) )
                            prev_res = results_taur[index_]
                            # print(index_, ":", prev_res.E_HFB)
                        res = prev_res
                                            
                        b20, pair = res.b20_isoscalar, getattr(res,PAIR_CONSTR,-1)
                        
                        BP[index_] = (b20, pair)
                        IJ.append(index_)
                        
                        Bmax, Bmin = max(Bmax, b20),  min(Bmin, b20)
                        Pmax, Pmin = max(Pmax, pair), min(Pmin, pair)
            
            
            # for k in results_taur.keys():
            #     if k not in key_order:
            #         print("key", k," missing in key_order")
            # _exportResult(results_taur, ## REMOVE
            #               'export_TRIz{}n{}_Tau{}.txt'.format(z,n, tail), key_order) 
            results_taur = deepcopy(aux)
            
            # results_taur = _sort_importData(results_taur)
        
        
        B = np.linspace(Bmin, Bmax, num=B_dims[1]-B_dims[0]+1, endpoint=True)
        P = np.linspace(Pmin, Pmax, num=P_dims[1]-P_dims[0]+1, endpoint=True)
        
        Bm, Pm = np.meshgrid(B, P)
        vals = np.zeros( (len(B), len(P)) )
        
        ## ================================================================= ##
        ##                                                                   ##
        ##                          ::  PLOTS   ::                           ##
        ##                                                                   ##
        ## ================================================================= ##
        #%% map the grid and the values 
        ## ================================================================= ##
        # z += 20
        # n += 20
        isot_name = '{}{}'.format(n+z, nuclei[z])
        
        import matplotlib.colors as mpl_c
                
        fig , ax = plt.subplots()
        x, y, txt, num_v = [],[],[], []
        # for k, val in BP.items():
        
        minVal, maxVal = 99999, -99999
        for index_, k in index_resuts.items():

            val = BP[index_]
            x.append( val[0] )
            y.append( val[1] )
            res = results_taur[k][1]
            v   = round(getattr(res, attr_), 3)
            
            minVal = min(minVal, v)
            maxVal = max(maxVal, v)
            vals[index_[0]-B_dims[0], index_[1]-P_dims[0]] = v
            
            num_v.append(v)
            txt.append(str(k))
        
        ax.scatter(x, y, c=num_v,  s=325)
        style = dict(size=10, color='black', rotation=20) # 'grey'
        for i in range(len(txt)):
            ax.text(x[i], y[i], txt[i], ha='center', **style)
        plt.colorbar(cm.ScalarMappable(
                norm=mpl_c.Normalize(vmin=min(num_v), vmax= max(num_v))), ax=ax)
        # plt.clim(min(num_v), max(num_v))
        # ax.axis('equal')
        plt.tight_layout()
        title_fig = 'BPTES_XYlist_'+isot_name+'_'+attr_+'_'+tail+'.pdf'
        
        plt.tight_layout()
        plt.title(attr_+"  {} {}  min={:6.3f}".format(z+n, nuclei[z], minVal))
        # plt.savefig(title_fig)
        plt.show()
        
        ## ================================================================= ##
        #%% MAIN FIGURE
        ## ================================================================= ##
        
        ### RESCALE THE VALUES
        
        for ir in range(len(B)):
            for it in range(len(P)):
                val_scale = vals[ir, it] - minVal*RESCALE_TES
                vals[ir, it] = val_scale     
        if RESCALE_TES:
            for k in BP.keys():
                res = results_taur[index_resuts[ k ]][1]
                setattr(res, attr_, getattr(res, attr_) - minVal)
        
        ## Plot
        
        fig, ax = plt.subplots() #subplot_kw={'projection': 'rectilinear'})
        cs1 = ax.contourf(Bm, Pm,  np.transpose(vals), levels=80)#) #
        levels = []
        cs2 = ax.contour(Bm, Pm, np.transpose(vals), 
                         colors=('w',), linewidths=(0.7,))#, levels=80) #
        
        # _Tticks = np.radians(np.arange(0, 61, 15))
        # ax.set_xticks(_Tticks)
        # ax.set_xlim([0, np.radians(60)])
        # ax.set_ylim([0, max(R)*1.01])
        
        # ax.tick_params(labelleft=True, labelright=True,
        #                labeltop =False, labelbottom=True)
        
        # ax.set_xticklabels(map(str, azimuths))   # Change the labels
        #plt.tight_layout()
        if RESCALE_TES:
            plt.colorbar(cm.ScalarMappable(
                norm=mpl_c.Normalize(vmin=0, vmax=maxVal - minVal)), ax=ax)
        else:
            plt.colorbar(cm.ScalarMappable(
                norm=mpl_c.Normalize(vmin=minVal, vmax=maxVal)), ax=ax)
        
        ax.clabel(cs2, fmt="%5.1f", colors='w', fontsize=10)
        
        plt.title(attr_+"  {} {}  min={:6.3f}".format(z+n, nuclei[z], 
                                                     minVal), loc='center')
        ax.set_xlabel("beta 20")
        ax.set_ylabel(PAIR_CONSTR)
        
        title_fig = 'BPTES_'+isot_name+'_'+attr_+'_'+tail+'.pdf'
        plt.tight_layout()
        # plt.savefig(title_fig)
        plt.show()
        
        ## ================================================================= ##
        #%% Check the triaxial plot with 2-D curves readed in j order
        ## ================================================================= ##
        fig, ax = plt.subplots()
        
        for j in range(P_dims[0], P_dims[1]+1, 4):
            x, y = [], []
            for i in range(B_dims[0], B_dims[1]+1):
                
                k = (j+P_dims[0]) + (i - B_dims[0])*(P_dims[1]-P_dims[0] + 1)
                
                res  = results_taur[k][1]
                # res  = results_taur[index_resuts[(i, j)]][1]
                
                x.append(res.b20_isoscalar)
                # x.append(getattr(res, Template.P_T00_J10))
                y.append(getattr(res, attr_))
                
            if len(x) > 0:
                ax.plot(x, y, '.-', label="{}={:5.3f}".format(PAIR_CONSTR, P[j]))
        
        plt.legend()
        ax.set_ylabel(attr_)
        plt.title(attr_+"  {} {}  min={:6.3f}".format(z+n, nuclei[z], 
                                                     minVal), loc='center')
        ax.set_xlabel('beta 20')
        # plt.legend()
        
        title_fig = 'Figure_BPTES_'+isot_name+'_'+attr_+'_'+tail+'.pdf'
        plt.tight_layout()
        # plt.savefig(title_fig)
        plt.show()


