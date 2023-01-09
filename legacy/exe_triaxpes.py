# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:53:55 2022

@author: Miguel
"""

from copy import deepcopy
from datetime import datetime
import os
import shutil
import subprocess

from exe_isotopeChain_taurus import DataTaurus
from exe_q20pes_taurus import Template, TemplateDDInp, template_DD_input
from exe_q20pes_taurus import _setDDTermInput, _printInput4Calculation, zipBUresults  # _exportResult (overwriten)
import math as mth
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


Template.b22 = 'b22'

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
Spatial one-body density      0
Discretization for x/r        0   0.00 
Discretization for y/theta    0   0.00 
Discretization for z/phi      0   0.00 

Iterative Procedure
-------------------
Maximum no. of iterations     {iters} 
Step intermediate wf writing  0
More intermediate printing    0
Type of gradient              {grad_type}
Parameter eta for gradient    {eta_grad:04.3f}E-00
Parameter mu  for gradient    {mu_grad:04.3f}E-00
Tolerance for gradient        {grad_tol:04.3}E-00

Constraints             
-----------
Force constraint N/Z          0
Constraint beta_lm            2
Pair coupling scheme          1
Tolerance for constraints     1.000E-08
Constraint multipole Q10      1   0.000
Constraint multipole Q11      1   0.000
Constraint multipole Q20      {b20}
Constraint multipole Q21      1   0.000
Constraint multipole Q22      {b22}
Constraint multipole Q30      0   0.000
Constraint multipole Q31      0   0.000
Constraint multipole Q32      0   0.000
Constraint multipole Q33      0   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      0   0.000
Constraint multipole Q42      0   0.000
Constraint multipole Q43      0   0.000
Constraint multipole Q44      0   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     0   0.000
Constraint pair P_T00_J1m1    0   0.000
Constraint pair P_T00_J1p1    0   0.000
Constraint pair P_T10_J00     0   0.000
Constraint pair P_T1m1_J00    0   0.000
Constraint pair P_T1p1_J00    0   0.000
Constraint field Delta        0   0.000
"""


nuclei = [
    #    Z  N

    # (8, 4),
    # (8, 6),
    # (8, 8),
    # (8, 10),
    # (8, 12),
    
    # (10, 6),
    # (10, 8),
    # (10, 10),
    # (10, 12),
    # (10, 14),
    
    # (12, 12),
    # (12, 10),
    # (12, 14),
    # (12, 16),
    
    (36, 34),
    (34, 36),
]

def _executeProgram(params, output_filename, bg_index, 
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
            
        str_q20 = '_'.join([str(round(x,3)) for x in bg_index]).replace('.','')
        
        _e = subprocess.call('./taurus_vap.exe < {} > {}'
                                  .format(DataTaurus.INPUT_FILENAME, 
                                          output_filename), 
                              shell=True,
                              timeout=43200) 
        res = DataTaurus(z, n, output_filename)        
        
        # move shit to the folder
        folder_dest = os.getcwd()+'/'+DataTaurus.BU_folder+'/'
        _e = subprocess.call('mv {} {}'.format(output_filename, 
                              folder_dest+output_filename
                              + '_Z{}N{}'.format(z,n)
                              +'_{}'.format(str_q20)),
                              shell=True,
                              timeout=8640) # 1 day timeout
        _e = subprocess.call('cp final_wf.bin '+folder_dest+
                             'seed_q{}_'.format(str_q20)+
                             '_Z{}N{}'.format(z,n)+'.bin', 
                             shell=True)
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
                          res.beta, res.gamma,
                          res.iter_max, params[Template.iterations]))
            #print(" >>", datetime.now().time())
    except Exception as e:
        print(" (1)>> EXCEP >>>>>>>>>>  ")
        print(" (1)>> current deform index =", str_q20)
        print(" 1>> OUTPUT FILE from TAURUS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        with open(output_filename, 'r') as f:
            text = f.read()
            print(text)
        print(" 1<< OUTPUT FILE from TAURUS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(" (1)> [",e.__class__.__name__,"]:", e, "<(1)")
        if res and res.E_HFB == None and not res.properly_finished:
            print(" (1)> the result is NULL (final_wf wasn't copied to initial_wf)")
        else:
            print(" 1> result not obtained, return None.")
        #print(res)
        print(" (1)<< EXCEP <<<<<<<<<<  ")
        return None
    
    return res


def _exportResult(results, path_, key_order):
    # TODO: modify to store the index of the value
    # key_order = [(0,0), (0,1), ...]
    data = []
    for key_ij in key_order:
        res = results.get(key_ij) #[]
        if res:
            line = res.getAttributesDictLike
            data.append(str(key_ij)+" = "+line+'\n')
    
    with open(path_, 'w+') as f:
        f.writelines(data)

# TODO: iteration form global minima
# as in the case for _set_deform_PES(res0), make a list for the x,y indexes for 
# the first result free minimum
def XYdeformPlane(i, j, L, x0=0, y0=0):
    x = x0 + (L * (0.5*j  + i))
    y = y0 + (0.5 * L * (3**0.5))*j
    return x, y

def BGdeform(x, y, key_ij=None, L=None):
    if key_ij and L:
        i, j = key_ij[0], key_ij[1]
        x, y = XYdeformPlane(i, j, L)
    
    b = (x**2  + y**2)**0.5
    g = 0.0
    if abs(x) > 1.0e-6:
        g = np.arctan2(y, x)
    
    return b, np.rad2deg(g)

def _set_deform_for_PES(res_0, b_max, N_max):
    """ 
    Deform plane and PES starting form the spherical.
    """
    L = b_max / (N_max-1)
    
    deform = []
    for j in range(N_max):
        for i in range(N_max):
            
            if i + j > N_max - 1:
                continue
            
            x, y = XYdeformPlane(i, j, L)
            b, g = BGdeform(x, y)
            
            deform.append(((i,j), (b,g)))
    
    return deform

 
def mainLinuxEvenBetaGamma(z, n, interaction, b_max=0.1, N_max=20,
                           voidDD_path=None):
    """ 
        Old process that sets an even single-evaluated step over b range
    voidDD_path is the equivalent of the DataTaurus.export_list_results for the 
    output of the final calculation
    """
    #%% Executing the process, run the list of isotopes
    #
    
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      beta,  gamma"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    if voidDD_path and os.path.exists(voidDD_path):
        os.remove(voidDD_path)
    
    b_base = 0.000
    g_base = 0.000
    results = {}
    results_voidStep = {}
    
    print(HEAD)
    # create a spherical/unbounded seed to proceed
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.z : z,     Template.n : n,
        Template.seed : 3,
        Template.grad_type : 1,
        Template.grad_tol  : 0.05, #0.005,
        Template.eta_grad : 0.03,
        Template.iterations : 500,
        Template.mu_grad  : 0.2,
        Template.b20  : "1 {:5.4}".format(b_base),
        Template.b22  : "1 {:5.4}".format(g_base),
    }
    RDIM_0, OMEG_0 = 8, 10
    _setDDTermInput(True, rdim=RDIM_0, omega=OMEG_0)
    
    print(" * first convergence (seed{}), interact={}"
            .format(kwargs[Template.seed], interaction))
    for iter_ in range(1, 4):
        res_0 = _executeProgram(kwargs, output_filename, (b_base,g_base))
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
    print("   ... done.\n")
      
    # ###
    deforms  = _set_deform_for_PES(res_0, b_max, N_max)
    ## WARNING! compromising start point
    _printInput4Calculation(N=N_max, dq = round(b_max/N_max, 4), b_LIMS=b_max, 
                            voidDD_path=voidDD_path, rdim=RDIM_0, omega=OMEG_0, 
                            **kwargs)  
    iter_  = 0
    j_prev = 0
    key_order = []
    
    _e = subprocess.call('cp ini_partial.bin initial_wf.bin', shell=True)
    for ij_key, deform in deforms:  
        iter_ += 1           
        save_partialWF = False
        key_order.append(ij_key)
        # copy the function from the spherical oblate side and save .
        if ij_key[1] != j_prev:
            _e = subprocess.call('cp ini_partial.bin initial_wf.bin', shell=True)
            j_prev = ij_key[1]
            save_partialWF = True
        
        b_def, g_def = deform[0], deform[1]
        
        ## deactivate DD
        kwargs = {
            Template.interaction : interaction, 
            Template.com : 1,   Template.read_red_hamil : 1,
            Template.z : z,     Template.n : n,
            Template.seed : 1,
            Template.grad_type : 1,
            Template.grad_tol  : 0.1, 
            Template.eta_grad : 0.05,
            Template.iterations : 700, # 700
            Template.mu_grad  : 0.08,
            Template.b20  : "1 {:5.4}".format(b_def),
            Template.b22  : "1 {:5.4}".format(g_def),
        }
        #res = _convergence_loop(kwargs, output_filename, b20, setDD=False)
        _setDDTermInput(True, rdim=8, omega=10)
        res = _executeProgram(kwargs, output_filename, (b_def,g_def),
                              save_final_wf=True)
        print("ij:[{} / {}] res: beta={} gamma={}".format(iter_, len(deforms), 
            getattr(res, 'beta',-1), getattr(res, 'gamma', None)))
        if res == None:
            continue # dont save empty result
        
        results[ij_key] = res   
        # intermediate print
        _exportResult(results, DataTaurus.export_list_results, key_order)
        ## SECOND PROCESS --------------------------------
        if voidDD_path == None:
            if save_partialWF:
                _e = subprocess.call('cp final_wf.bin ini_partial.bin', shell=True)
            continue
        
        ##do a void step to activate DD with no rearrangement
        _setDDTermInput(True, rdim=10, omega=10)
        kwargs = {
            Template.interaction : interaction, 
            Template.com : 1,   Template.read_red_hamil : 0,
            Template.z : z,     Template.n : n,
            Template.seed : 1,
            Template.grad_type : 0,
            Template.grad_tol  : 0.01,
            Template.eta_grad : 0.005,
            Template.iterations : 150,
            Template.mu_grad  : 0.1,
            Template.b20  : "1 {:5.4}".format(b_def),
            Template.b22  : "1 {:5.4}".format(g_def),
        }
        _setDDTermInput(True, rdim=8, omega=10)
        res2 = _executeProgram(kwargs, output_filename+'_VS_', (b_def,g_def),
                               save_final_wf=False)
        print("ij:[{} / {}] res2: beta={} gamma={}".format(iter_, len(deforms), 
            getattr(res, 'beta',-1), getattr(res, 'gamma', None)))
        print()
        
        if res2 == None:
            continue # dont save empty result
        
        if save_partialWF:
            # save also here, the initial wf will be preciser.
            _e = subprocess.call('cp final_wf.bin ini_partial.bin', shell=True)
        
        results_voidStep[ij_key] = res2        
        # intermediate print
        _exportResult(results_voidStep, voidDD_path, key_order) 
    
    # ## ------ end exec.  -----
    _exportResult(results, DataTaurus.export_list_results, key_order)
    print("   ** generate File 1st convergence in:", DataTaurus.export_list_results)
    if results_voidStep:
        _exportResult(results_voidStep, voidDD_path)
        print("   ** generate File VoidStep in:", voidDD_path)


#%% Triaxial ploting adjusting methods

def isPointInTriangle(x,y, x0,y0, x1,y1, x2,y2):
    """ method from https://mathworld.wolfram.com/TriangleInterior.html """
    v1 = (x1-x0, y1-y0)
    v2 = (x2-x0, y2-y0)
    
    C = (v1[0]*v2[1]) - (v1[1]*v2[0])
    A = ((x*v2[1]) - (y*v2[0])) - ((x0*v2[1] - y0*v2[0]))
    B = ((x*v1[1]) - (y*v1[0])) - ((x0*v1[1] - y0*v1[0]))
    
    A /=  C 
    B /= -C
    
    # print("a ={:18.15f}\nb ={:18.15f}\nc ={:18.15f}\n  ={:18.15f}"
           # .format(A,B,C,A+B))
    if A > 0.0 and B > 0.0:
        if (A + B) <= 1.0000000000000000:
            return True
    return False

def getInterpolatedValue(x,y, x1,y1,z1, x2,y2,z2, x3,y3,z3):
    """ Equations of the plane defined by three points (1,2,3), returns z """
    x21 = x2 - x1
    y21 = y2 - y1
    z21 = z2 - z1
    
    x31 = x3 - x1
    y31 = y3 - y1
    z31 = z3 - z1
    
    Cx = (y21*z31) - (y31*z21)
    Cy = (x21*z31) - (x31*z21)
    Cz = (x21*y31) - (x31*y21)
    
    z  = (((Cy * y) - (Cx * x) + ((Cx*x1) - (Cy*y1))) / Cz) + z1
    
    return z

#%% main process
b_lenght = 1.8
z = 36  #12 #6 #
n = 34 #12 #8 #

tail = ''
# tail = 'D1SnoR'
# tail = 'D1S_Fixed'
tail = 'D1S'
    
nuclei_name = {2:'He', 4:'Be', 6:'C', 8:'O', 10:'Ne', 12:'Mg', 14:'Si', 16:'S', 18:'Ar', 20:'Ca',
               34: 'Se', 36: 'Kr', 38:'Sr'}
if __name__ == '__main__':
    
    interaction = 'D1S_t0_SPSDPF' #'D1S_t0_SPSDPF' # 'B1_SPSD' #'0s0p1s0d'  #
    interaction = 'D1S_t0_MZ6'
    # interaction = 'D1S_Mg24_SPSD'
        
    if not os.getcwd().startswith('C:'):    #
        assert "voidDD" not in tail, "[ERROR] [STOP] 'voidDD' in tail for exporting name!"
        print()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(' Running PES with Taurus_VAP, hamilonian:', interaction)
        print('  !!! CHECK, CHECK MZ:', interaction.upper())
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print("GLOBAL START:", datetime.now().time())
        print()
        
        for z, n in nuclei:
            print("Start mainLinux: ", datetime.now().time())
            print("PES for Z={} N={}".format(z,n))
            DataTaurus.export_list_results = "export_TPESz{}n{}Taurus{}.txt".format(z,n, tail)
            
            mainLinuxEvenBetaGamma(z,n, interaction, 0.6, 16)
            
            zipBUresults(z, n, interaction, 'beta', 'gamma')
            print("End mainLinux: ", datetime.now().time())
            
    else:
        RESCALE_TES = True
        z = 36  #12 #6 #
        n = 34  #12 #8 #
        
        results_taur = []
        index_resuts = {}
        #from exe_isotopeChain_axial import DataAxial
        PESfolder = 'BU_triaxial/'
        PESfolder = 'BU_triaxial/SPSD/'
        PESfolder = 'BU_triaxial/SPSDPF/'
        PESfolder = 'BU_triaxial/MZ5/'
        
        DataTaurus.export_list_results = "export_TPESz{}n{}Taurus{}.txt".format(z,n, tail)
        import_file_Tau = PESfolder + DataTaurus.export_list_results        
        
        ## get results from export "csv" 
        RT = dict()
        IJ = []
        Rmax, Tmax = 0.0, 0.0
        Rmin, Tmin = 0.0, 0.0
        print("IMPORTING: ", import_file_Tau)
        if os.path.exists(import_file_Tau):
            N_max = 0
            with open(import_file_Tau, 'r') as f:
                data = f.readlines()
                for line in data:
                    index_, line = line.split(' = ')
                    res = DataTaurus(None, None, None, True)
                    res.setDataFromCSVLine(line)
                    index_ = tuple([int(x) for x in index_[1:-1].split(',')])
                    results_taur.append((index_, res))
                    
                    # print(index_, ':', res.E_HFB,',')
                    N = max(index_)
                    if N > N_max:
                        N_max = N
            
            results_taur = dict(results_taur)
            aux = []
            # TODO: sort the data 
            # TODO: 
            prev_res = None
            key_order = []
            for j in range(N_max+1):
                for i in range(N_max+1):
                    if i + j > N_max:
                        continue
                    elif (i, j) == (N_max, 0) or (i, j) == (0, N_max):
                        print(i, j)
                        pass
                    
                    index_ = (i, j)
                    key_order.append(index_) ## REMOVE
                    if index_ not in results_taur:
                        aux.append( (index_, prev_res) )
                    else:
                        aux.append( (index_, results_taur[index_]) )
                        prev_res = results_taur[index_]
                        # print(index_, ":", prev_res.E_HFB)
                    res = aux[-1][1]
                    index_resuts[index_] = len(aux) - 1
                    
                    r, t = res.beta, res.gamma
                    if j == 0:
                        if abs(t-360) < 0.01 or abs(t-180) < 0.01:
                            t = 0.0
                    t = np.deg2rad(t)
                    # RT[index_] = r + 1j*t 
                    RT[index_] = (r, t)
                    IJ.append(index_)
                    
                    Rmax = max(Rmax, r)
                    Tmax = max(Tmax, t)
                    Rmin = min(Rmin, r)
                    Tmin = min(Tmin, t)
            
            # for k in results_taur.keys():
            #     if k not in key_order:
            #         print("key", k," missing in key_order")
            _exportResult(results_taur, ## REMOVE
                          'export_TRIz{}n{}_Tau{}.txt'.format(z,n, tail), key_order) 
            results_taur = deepcopy(aux)
            
            # results_taur = _sort_importData(results_taur)
        
        # TODO: plot        
        attr_ = 'E_HFB'
        # attr_ = 'kin'
        # attr_ = 'pair_pp'
        # attr_ = 'beta'
        # attr_ = 'gamma'
        # attr_ = 'iter_time_seconds'
        
        ## Reduce the space to search R and T to be fully inside the Triangular Data
        Rmax *= min(1, .866) #85
        Rmin += 0.001
        Tmax -= 0.001
        Tmin += 0.001
        
        
        R = np.linspace(Rmin, Rmax, num=N_max+1, endpoint=True)
        T = np.linspace(Tmin, Tmax, num=N_max+1, endpoint=True)
        
        Rm, Tm = np.meshgrid(R, T)
        vals = np.zeros( (len(R), len(T)) )
        
        ## read all the points in R,T
        maxVal = -9999999
        minVal =  9999999
        x, y = 0, 0
        for ir in range(len(R)):
            print(' i{:2},{:2}: rt({:5.4f}  {:5.2f})  xy({:4.3f} {:4.3f}) ----------------'
                       .format(ir, len(T), R[ir], T[-1], x, y))
            for it in range(len(T)):
                
                x = Rm[ir, it] * np.cos(Tm[ir, it])
                y = Rm[ir, it] * np.sin(Tm[ir, it])
                # print(' i{:2},{:2}: rt({:5.4f}  {:5.2f})  xy({:4.3f} {:4.3f}) ----------------'
                #        .format(ir, it, R[ir], T[it], x, y))
                ## search the i, j of the lower triangle point
                stop_ = False
                for j1 in range(N_max+1):
                    if stop_:continue
                    for i1 in range(N_max):
                        if stop_:continue
                        if j1 + i1 > N_max-1:
                            continue
                        
                        # print("i1,j1=", i1, j1)
                        for i2_perm in (0, 1):
                            i2 = max(0,i1 - ((-1)**i2_perm))
                            j2 = j1 + (1 - i2_perm)
                            if j2 + i2 > N_max: 
                                # print("i2,j2=", i2, j2)
                                continue
                            
                            i3, j3 = i1,  j1+1
                            
                            rt0, rt1, rt2 = RT[(i1,j1)], RT[(i2,j2)], RT[(i3,j3)]
                            
                            x0 = rt0[0] * np.cos(rt0[1])
                            y0 = rt0[0] * np.sin(rt0[1])
                            x1 = rt1[0] * np.cos(rt1[1])
                            y1 = rt1[0] * np.sin(rt1[1])
                            x2 = rt2[0] * np.cos(rt2[1])
                            y2 = rt2[0] * np.sin(rt2[1])
                            
                            if not isPointInTriangle(x, y, x0, y0, x1, y1, x2, y2):
                                continue
                            
                            val0 = getattr(results_taur[index_resuts[(i1, j1)]][1], attr_)
                            val1 = getattr(results_taur[index_resuts[(i2, j2)]][1], attr_)
                            val2 = getattr(results_taur[index_resuts[(i3, j3)]][1], attr_)
                            
                            args = (x, y, x0, y0, val0, x1, y1, val1, x2, y2, val2)
                            val = getInterpolatedValue(*args)
                            
                            vals[ir, it] = val
                            
                            maxVal = max(maxVal, val)
                            minVal = min(minVal, val)
                            stop_ = True
                            # input("coordinate FOUND")
                        # print("==========end perm =============")
        
                if not stop_:
                    print(" coordinate NOT FOUND")
                    # vals[ir, it] = (vals[ir-1,it-1]+vals[ir,it-1]+vals[ir-1,it])/3
                    vals[ir, it] = maxVal
        
        ### RESCALE THE VALUES
        if RESCALE_TES:
            for ir in range(len(R)):
                for it in range(len(T)):
                    val_scale = vals[ir, it] - minVal
                    vals[ir, it] = val_scale     
        
            for k in RT.keys():
                res = results_taur[index_resuts[ k ]][1]
                setattr(res, attr_, getattr(res, attr_) - minVal)
        
        
        ## ================================================================= ##
        ##                                                                   ##
        ##                          ::  PLOTS   ::                           ##
        ##                                                                   ##
        ## ================================================================= ##
        #%% map the grid and the values 
        ## ================================================================= ##
        isot_name = '{}{}'.format(n+z, nuclei_name[z])
        
        import matplotlib.colors as mpl_c
                
        fig , ax = plt.subplots()
        x, y, txt, num_v = [],[],[], []
        for k, val in RT.items():
            
            x.append( val[0] * np.cos(val[1]) )
            y.append( val[0] * np.sin(val[1]) )
            res = results_taur[index_resuts[ k ]][1]
            v   = round(getattr(res, attr_), 3)
            
            num_v.append(v)
            txt.append(str(k))
        
        xR, yR = [], []
        for ir in range(len(R)):
            for it in range(len(T)):
                xR.append(R[ir]*np.cos(T[it]))
                yR.append(R[ir]*np.sin(T[it]))
        
        ax.scatter(x, y, c=num_v,  s=325)
        ax.scatter(xR,yR, c='red', s=3)
        style = dict(size=10, color='gray', rotation=20)
        for i in range(len(txt)):
            ax.text(x[i], y[i], txt[i], ha='center', **style)
        plt.colorbar(cm.ScalarMappable(
                norm=mpl_c.Normalize(vmin=min(num_v), vmax= max(num_v))), ax=ax)
        # plt.clim(min(num_v), max(num_v))
        ax.axis('equal')
        plt.tight_layout()
        title_fig = 'TPES_XYlist_'+isot_name+'_'+attr_+'_'+tail+'.pdf'
        
        plt.tight_layout()
        plt.title(attr_+"  {} {}  min={:6.3f}".format(z+n, nuclei_name[z], minVal))
        # plt.savefig(title_fig)
        plt.show()
        
        ## ================================================================= ##
        #%% MAIN PIE FIGURE
        ## ================================================================= ##
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        cs1 = ax.contourf(Tm, Rm, vals, levels=80)
        levels = []
        cs2 = ax.contour(Tm, Rm, vals, 
                         colors=('w',), linewidths=(0.7,), levels=80)
        
        _Tticks = np.radians(np.arange(0, 61, 15))
        ax.set_xticks(_Tticks)
        ax.set_xlim([0, np.radians(60)])
        ax.set_ylim([0, max(R)*1.01])
        
        ax.tick_params(labelleft=True, labelright=True,
                       labeltop =False, labelbottom=True)
        
        # ax.set_xticklabels(map(str, azimuths))   # Change the labels
        #plt.tight_layout()
        if RESCALE_TES:
            plt.colorbar(cm.ScalarMappable(
                norm=mpl_c.Normalize(vmin=0, vmax=maxVal - minVal)), ax=ax)
        else:
            plt.colorbar(cm.ScalarMappable(
                norm=mpl_c.Normalize(vmin=minVal, vmax=maxVal)), ax=ax)
        
        ax.clabel(cs2, fmt="%5.1f", colors='w', fontsize=10)
        
        plt.title(attr_+"  {} {}  min={:6.3f}".format(z+n, nuclei_name[z], 
                                                     minVal), loc='left')
        
        title_fig = 'TPES_'+isot_name+'_'+attr_+'_'+tail+'.pdf'
        plt.tight_layout()
        # plt.savefig(title_fig)
        plt.show()
        
        ## ================================================================= ##
        # #%% Check the triaxial plot with 2-D curves readed in j order
        ## ================================================================= ##
        fig, ax = plt.subplots()
        
        x, y = [], []
        for i in range(N_max+1):
            
            res_obl  = results_taur[index_resuts[(0, i)]][1]
            res_pro  = results_taur[index_resuts[(i, 0)]][1]
            
            x.insert(0, -res_obl.beta)
            x.append(res_pro.beta)
            y.insert(0, getattr(res_obl, attr_))
            y.append(getattr(res_pro, attr_))
            
        if len(x) > 0:
            ax.plot(x, y, 'r.-')
        ax.set_ylabel(attr_)
        plt.title(attr_+"  {} {}  min={:6.3f}".format(z+n, nuclei_name[z], 
                                                     minVal), loc='center')
        ax.set_xlabel('beta 2(gamma=0º)')
        # plt.legend()
        
        title_fig = 'Figure_axialTPES_'+isot_name+'_'+attr_+'_'+tail+'.pdf'
        plt.tight_layout()
        # plt.savefig(title_fig)
        plt.show()
        
        
        # ## ================================================================= ##
        # ## Imshow the values with only the indexes
        # ## ================================================================= ##
        # fig, ax = plt.subplots()
        # I = [i for i in range(N_max)]
        # J = [i for i in range(N_max)]
        # I, J = np.meshgrid(I, J)
        # imsh_vals = np.zeros((N_max, N_max)) - 0.5*abs(maxVal+minVal)
        
        # for j in range(N_max):
        #     x, y = [], []
        #     for i in range(N_max):
        #         if i + j > N_max:
        #             continue
        #         imsh_vals[i, j] = getattr(results_taur[index_resuts[(i, j)]][1], 
        #                                   attr_)
                
        # im = plt.imshow(imsh_vals)
        # cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        # cb.set_ticks([])
        # ax.set_ylim([-0.5, N_max+0.5])
        # plt.show()
        
                    # print the pdf images and the Latex-figure- Template

        
        
        latex_fig = "\\subsubsection{"+isot_name+", Figure \\ref{fig:"+isot_name+"_"+tail+"_MZ2_triaxD1S}}\n" +\
"\\begin{figure}\n"+\
"    \\centering\n"+\
"    \\subfloat {{\\includegraphics[width=120mm]{TriPES_SPSD/TPES_"+isot_name+"_"+attr_+"_"+tail+".pdf} }}\n"+\
"    \\qquad\n"+\
"    \\subfloat[\\centering Interpolation map for the previous chart.]\n"+\
"    {{\\includegraphics[width=80mm]{TriPES_SPSD/TPES_XYlist_"+isot_name+"_"+attr_+"_"+tail+".pdf} }}\n"+\
"    \\qquad\n"+\
"    \\subfloat[\\centering Axial profile of the surface.]\n"+\
"    {{\\includegraphics[width=100mm]{TriPES_SPSD/Figure_axialTPES_"+isot_name+"_"+attr_+"_"+tail+".pdf} }}\n"+\
"    \\caption{Calculation for "+isot_name+" on the SPSD shell space. b length = "+str(b_lenght)+" fm}\n"+\
"    \\label{fig:"+isot_name+"_D1S_MZ2_triaxD1S}\n"+\
"\\end{figure}\n"
        
        print('\n')
        print()
        print(latex_fig)
        
        print()
        
            
        