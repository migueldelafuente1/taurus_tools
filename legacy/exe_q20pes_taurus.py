"""
Created on Fri Mar  4 19:28:46 2022

@author: Miguel
"""

class Template:
    interaction = 'interaction'
    com         = 'com'
    read_red_hamil = 'red_hamil' 
    z   = 'z'
    n   = 'n'
    seed= 'seed'
    b20 = 'b20'
    msr = 'msr' 
    grad_type  = 'grad_type'
    grad_tol   = 'grad_tol'
    eta_grad   = 'eta_grad' 
    mu_grad    = 'mu_grad'
    iterations = 'iters'

class TemplateDDInp:
    DD_term  = 'DD_on'
    Rea_term = 'REA_on'
    rdim     = 'rdim'
    omega    = 'omega'
    Explicit = 'Explicit'
    

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
Constraint multipole Q22      1   0.000
Constraint multipole Q30      0   0.000
Constraint multipole Q31      1   0.000
Constraint multipole Q32      1   0.000
Constraint multipole Q33      0   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      1   0.000
Constraint multipole Q42      1   0.000
Constraint multipole Q43      0   0.000
Constraint multipole Q44      0   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     1   0.000
Constraint pair P_T00_J1m1    1   0.000
Constraint pair P_T00_J1p1    1   0.000
Constraint pair P_T10_J00     0   0.000
Constraint pair P_T1m1_J00    0   0.000
Constraint pair P_T1p1_J00    0   0.000
Constraint field Delta        0   0.000
"""

template_DD_input = """* Density dep. Interaction:    ------------
eval_density_dependent (1,0)= {DD_on}
eval_rearrangement (1,0)    = {REA_on}
eval_explicit_fieldsDD (1,0)= {Explicit}
t3_DD_CONST [real  MeV]     = 1.390600d+03    !1.000000d+03    !
x0_DD_FACTOR                = 1.000000d+00
alpha_DD                    = 0.333333d+00    !1.000000d+00    !
* Integration parameters:      ------------
*  0 trapezoidal, 1 Gauss-Legendre, 2 Gauss-Laguerre(r)/Legendre, 3 Laguerre-Lebedev
integration_method (0,1,2,3)= 3
export_density (1, 0)       = 0
r_dim                       = {rdim}
Omega_Order                 = {omega}
THE_grid                    = 10
PHI_grid                    = 10
R_MAX                       = 08.500000d+00
eval full Val.Space (0,1)   = 1
* Integration parameters:      ------------
"""

from collections import OrderedDict
from datetime import datetime
import os
import shutil
import subprocess

from exe_isotopeChain_axial import DataAxial
from exe_isotopeChain_taurus import DataTaurus
import math as mth
import matplotlib.pyplot as plt
import numpy as np


nucleus = [
#    Z  N
    # (2, 2), 
    # (2, 4),
    
    # (4, 4), 
    # (4, 6),
    #
    # (6, 6), 
    # (6, 8),
    # (6, 10),
    #
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
    #
    # (12, 8),
    # (12, 10),
    (12, 12),
    # (12, 14),
    # (12, 16),
    #
    # (14, 8),
    # (14, 10),
    # (14, 12),
    # (14, 14),
    # (14, 16),
    #
    # (16, 12),
    # (16, 14),
    # (16, 16),
    # (16, 18),
    # (16, 20),
    
    # (38, 40),
    # (40, 38),
    
    # (36, 34),
    # (34, 36),
]

# nucleus = [(8, n) for n in range(6, 15, 2)]

#%% main
def zipBUresults(z,n,interaction, *args):
    """
    This method export BU_folder results and outputs into a .zip, adding an 
    extension for the times the script result and zip export has been used in 
    the directory.
    """
    
    buzip = "BU_z{}n{}-{}_{}".format(z,n,interaction,'-'.join(args))
    current_zipfiles = filter(lambda x: x.endswith('.zip'), os.listdir('.'))
    count_buzips = list(filter(lambda x: x.startswith(buzip), current_zipfiles))
    
    buzip += '_{}.zip'.format(len(count_buzips))
    
    order = 'zip -r {} {}'.format(buzip, DataTaurus.BU_folder)
    try:
        _e = subprocess.call(order, shell=True)
    except BaseException as be:
        print("[ERROR] zipping of the BUresults cannot be done:: $",order)
        print(">>>", be.__class__.__name__, ":", be)

def recoverResults(z, n):
    
    output_filename = DataTaurus.output_filename_DEFAULT
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    
    results = []
    for b20_const in np.linspace(-0.2, 0.2, num=21, endpoint=True):
        output_ = os.getcwd()+'/'+DataTaurus.BU_folder \
                                     +'/'+output_filename \
                                     + '_Z{}N{}'.format(z,n) \
                        +'_{}'.format(str(int(1000*b20_const)).replace('-','neg'))
        
        res = DataTaurus(z, n, output_)
        results.append(res)
    
    data = []
    for res in results:
        # line = ', '.join([k+' : '+str(v) for k,v in res.__dict__.items()])
        line = res.getAttributesDictLike
        data.append(line+'\n')
        
    with open(DataTaurus.export_list_results, 'a+') as f:
        f.writelines(data)

def _executeProgram(params, output_filename, q20_const, 
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
        
        _e = subprocess.call('./taurus_vap.exe < {} > {}'
                                  .format(DataTaurus.INPUT_FILENAME, 
                                          output_filename), 
                              shell=True,
                              timeout=43200) # 12 timeout
        res = DataTaurus(z, n, output_filename)        
        
        # move shit to the folder
        str_q20 = str(int(1000*q20_const)).replace('-','_')
        folder_dest = os.getcwd()+'/'+DataTaurus.BU_folder+'/'
        _e = subprocess.call('mv {} {}'.format(output_filename, 
                              folder_dest+output_filename
                              + '_Z{}N{}'.format(z,n)
                              +'_{}'.format(str_q20)),
                              shell=True) 
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
            print(" {:2} {:2}  ({})    {:9.4f}  {:9.4f}  {:7.4f}  {:6.4f}={:6.2f} [{}/{}: {}']"
                  .format(z, n, status_fin, res.E_HFB, res.kin, res.pair, 
                          res.b20_isoscalar, getattr(res,'q20_isoscalar',1e3) * 1.5853309,
                          res.iter_max, params[Template.iterations], 
                          res.iter_time_seconds//60))
            #print(" >>", datetime.now().time())
    except Exception as e:
        print("\n 1>> EXCEP (exe_q20pes_taurus._executeProgram) >>>>>>>  ")
        print(" 1>> current b20 =", q20_const)
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

def _setDDTermInput(set_on, rdim=10, omega=10, rea_on=True):
    # create a default DD_params input
    if set_on:
        rea_on = 1 * rea_on
        kwargsdd = {TemplateDDInp.DD_term: 1, 
                    TemplateDDInp.Rea_term: rea_on,
                    TemplateDDInp.rdim : rdim,
                    TemplateDDInp.omega: omega,
                    TemplateDDInp.Explicit: 0}
        text = template_DD_input.format(**kwargsdd)
        with open('input_DD_PARAMS.txt', 'w+') as f:
            f.write(text)
    else:
        kwargsdd = {TemplateDDInp.DD_term: 0, 
                    TemplateDDInp.Rea_term: 0,
                    TemplateDDInp.rdim : rdim,
                    TemplateDDInp.omega: omega,
                    TemplateDDInp.Explicit: 0}
        text = template_DD_input.format(**kwargsdd)
        with open('input_DD_PARAMS.txt', 'w+') as f:
            f.write(text)

def _set_deform_for_PES(res_0, b_min=-0.3,  b_max=0.3, N = 20):
    """
    :res_0 <DataTaurus object>
        Set an evenly spaced grid, dividing in "oblate" for points to the left 
    of a b_20 minumum and "prolate" to the right.
        In case the seed minumum is outside the range, the old range is shifted 
    and centered to the new b20.
    """
    
    N = 2 * (N // 2) # only even number N/2
    b_range = b_max - b_min
    assert b_min < b_max, \
        "b_max[{}] needs to be extricly greater than b_min[{}]!".format(b_max, b_min)
    
    dq = b_range / N
    dq_decimals = int(mth.ceil(abs(np.log10(dq)))) + 1 # 2 significative decimals
    dq = round(dq, dq_decimals)
    
    b = getattr(res_0, 'b20_isoscalar', 0.0) # default 0.0
    
    if b > b_max or b < b_min:
        b_max = b + (b_range / 2) # * abs(b_max) / abs(b_max))
        b_min = b - (b_range / 2) #* abs(b_min) /abs(b_min))
        
        print("Min/Max :: ", b_min, b_max,  b_max - b_min)
        
    # b = round(b_min + (dq * ((b - b_min) // dq)), dq_decimals)
    # print("b1=", b1," to ",b)
    
    total_def = np.linspace(b_min, b_max, num=N, endpoint=True)
    deform_prolate = list(filter(lambda x: x > b, total_def))
    deform_oblate  = list(filter(lambda x: x <= b, total_def))
    deform_oblate.append(b)
    deform_oblate.reverse()
    Npro = len(deform_prolate)
    Nobl = N - Npro
    
    return deform_oblate, deform_prolate

def mainLinuxDDFixed(z, n, interaction):
    #%% Executing the process, run the list of isotopes
    #
    
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    
    results = []
    
    print(HEAD)
    deform_prolate = np.linspace(0.0, 25.0, num=25, endpoint=True)
    deform_oblate  = np.linspace(0.0,-25.0, num=25, endpoint=True) #18,
    constr_mr2 = '0   0.0000'
    for i_deform, deform in enumerate((deform_oblate, deform_prolate)):
        # create a default DD_params input
        _setDDTermInput(False, rdim=12, omega=15)  # for DD fixed
        
        # create a spherical seed to proceed
        kwargs = {
            Template.interaction : interaction, 
            Template.com : 1,   Template.read_red_hamil : 0,
            Template.z : z,     Template.n : n,
            Template.seed : 3,
            Template.grad_type : 0,
            Template.eta_grad : 0.005,
            Template.iterations : 2000,
            Template.mu_grad  : 0.2,
            Template.b20  : "0 {:5.4}".format(0.000),
        }
        print(" * first convergence (seed{}), interact={}"
                .format(kwargs[Template.seed], interaction))
        for iter_ in range(1, 4):
            res_0 = _executeProgram(kwargs, output_filename, 0.0)
            if res_0.properly_finished:
                break
            else:
                if iter_ == 3:
                    print("[ERROR], after 4 tries the calculation STOP for", z, n)
                    return
            kwargs[Template.eta_grad]   -= 0.007 * iter_
            kwargs[Template.eta_grad]    = max(kwargs[Template.eta_grad], 0.001)
            kwargs[Template.iterations] += 150 * (iter_ + 1)
            print(" [WARNING] 1st step non converged, next try:", iter_, kwargs[Template.eta_grad])
        
        for q20_const in deform:
            ## activate DD 
            _setDDTermInput(False, rdim=3, omega=4) # for DD fixed
            
            kwargs = {
                Template.interaction : interaction, 
                Template.com : 1,   Template.read_red_hamil : 1,
                Template.z   : z,   Template.n   : n,
                Template.seed : 1,
                Template.grad_type : 0,
                Template.eta_grad : 0.005,
                Template.iterations : 1000,
                Template.mu_grad  : 0.2,
                Template.b20  : "1 {:5.4}".format(q20_const),
            }
            res = _executeProgram(kwargs, output_filename, q20_const,
                                  print_result=True)
            if res == None:
                continue # dont save empty result
            
            # # do a void step to activate DD with no rearrangement
            # _setDDTermInput(True, rdim=10, omega=10)
            # kwargs = {
            #     Template.interaction : interaction, 
            #     Template.com : 1, Template.read_red_hamil : 1,
            #     Template.z   : z, Template.n   : n,
            #     Template.seed : 1,
            #     Template.grad_type : 0,
            #     Template.eta_grad : 0.005,
            #     Template.iterations : 100,
            #     Template.mu_grad  : 0.1,
            #     Template.b20  : "1 {:5.4}".format(q20_const),
            # }
            # res = _executeProgram(kwargs, output_filename, q20_const)
            
            if res == None:
                continue # dont save empty result
            
            if i_deform == 0:
                results.insert(0, res)
            else:
                results.append(res)   
            
                     
    
    # ## ------ end exec.  -----
    data = []
    for res in results:
        line = res.getAttributesDictLike
        data.append(line+'\n')
        
    with open(DataTaurus.export_list_results, 'a+') as f:
        f.writelines(data)

def _convergence_loop(params, output_filename, b20,
                      save_final_wf=True, setDD=True):
    """ 
    Function to loop until a converged function come out."""
    # _setDDTermInput(setDD, rdim=10, omega=7)
    eta_l = Template.eta_grad
    i = 0
    while i < 4:
        i += 1
        res = _executeProgram(params, output_filename, b20, 
                              print_result=True, save_final_wf=save_final_wf, 
                              force_converg = True)
        
        if res and res.properly_finished:
            # due force_converg=TRUE, wf_ was saved
            return res
        else:
            # reduce the eta_ gradient a 30 % and increase  the iterations
            params[eta_l] = round(params[eta_l] * 0.7, 3)
            params[Template.iterations] += 100 
            if i < 4:
                print("    ** Repeating the iteration, new eta=", params[eta_l])
    print(" [WARNING] Could'nt properly converge the solution by reducing eta.")
    return res

def _printInput4Calculation(**kwargs):
    
    print('\n%%% STARING CALCULATION PARAMS %%%%%%%%%%%%%%%%%%%%%%%')
    for p, val in kwargs.items():
        print(p.upper(),"  :: ", val)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    
def mainLinuxEvenlyDeform(z, n, interaction, b_min=-0.1, b_max=0.1, N_max=50,
                          voidDD_path=None):
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
    if voidDD_path and os.path.exists(voidDD_path):
        os.remove(voidDD_path)
    
    b20_base = 0.000 #b_min + 0.01 # 
    results = []
    results_voidStep = []
    
    print(HEAD)
    # create a spherical/unbounded seed to proceed
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.z : z,     Template.n : n,
        Template.seed : 1,
        Template.grad_type : 1,
        Template.grad_tol  : 0.005,
        Template.eta_grad : 0.03,
        Template.iterations : 000,
        Template.mu_grad  : 0.2,
        Template.b20  : "0 {:5.4}".format(b20_base),
    }
    RDIM_0, OMEG_0 = 12, 14
    _setDDTermInput(True, rdim=RDIM_0, omega=OMEG_0)
    
    print(" * first convergence (seed{}), interact={}"
            .format(kwargs[Template.seed], interaction))
    for iter_ in range(1, 4):
        res_0 = _executeProgram(kwargs, output_filename, 0.0)
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
    _e = subprocess.call('cp final_wf.bin initial_Spheric.bin', shell=True)
    print("   ... done.")
      
    # ###
    deform_oblate, deform_prolate  = _set_deform_for_PES(res_0,b_min,b_max,N_max)
    ## WARNING! compromising start point
    _printInput4Calculation(dq=deform_oblate[1]-deform_oblate[0], N=N_max,
                            b_LIMS=(deform_oblate[-1],deform_prolate[-1]), 
                            voidDD_path=voidDD_path, 
                            rdim=RDIM_0, omega=OMEG_0, **kwargs)  
    
    for i_deform, deform in enumerate((deform_oblate, deform_prolate)):             
        # copy it.
        _e = subprocess.call('cp initial_Spheric.bin initial_wf.bin', shell=True)
        
        for b20 in deform:
            ## deactivate DD
            kwargs = {
                Template.interaction : interaction, 
                Template.com : 0,   Template.read_red_hamil : 0,
                Template.z : z,     Template.n : n,
                Template.seed : 1,
                Template.grad_type : 0,
                Template.grad_tol  : 0.02,
                Template.eta_grad : 0.005,
                Template.iterations : 700,
                Template.mu_grad  : 0.1,
                Template.b20  : "1 {:5.4}".format(b20),
            }
            #res = _convergence_loop(kwargs, output_filename, b20, setDD=False)
            _setDDTermInput(True, rdim=4, omega=4)
            res = _executeProgram(kwargs, output_filename, b20,
                                   save_final_wf=True)
            print("res.b20_isoscalar =", getattr(res, 'b20_isoscalar', None))
            if res == None:
                continue # dont save empty result
            if i_deform == 0:
                results.insert(0, res) #grow in order [-.5, -.4, ..., .0,..., +.4, +.5] 
            else:
                results.append(res)   
            
            ## SECOND PROCESS --------------------------------
            if voidDD_path == None:
                continue
            
            ##do a void step to activate DD with no rearrangement
            _setDDTermInput(True, rdim=10, omega=10)
            kwargs = {
                Template.interaction : interaction, 
                Template.com : 0,   Template.read_red_hamil : 0,
                Template.z : z,     Template.n : n,
                Template.seed : 1,
                Template.grad_type : 0,
                Template.grad_tol  : 0.01,
                Template.eta_grad : 0.005,
                Template.iterations : 150,
                Template.mu_grad  : 0.1,
                Template.b20  : "1 {:5.4}".format(b20),
            }
            print("2:", output_filename, b20)
            _setDDTermInput(True, rdim=8, omega=10)
            res2 = _executeProgram(kwargs, output_filename+'_VS_', b20,
                                   save_final_wf=False)
            print("res.b2_isoscalar =", getattr(res2, 'b20_isoscalar', None))
            print()
            
            if res2 == None:
                continue # dont save empty result
            if i_deform == 0:
                results_voidStep.insert(0, res2)
            else:
                results_voidStep.append(res2)
            
            # intermediate print
            _exportResult(results, DataTaurus.export_list_results)
            if voidDD_path != None:
                _exportResult(results_voidStep, voidDD_path) 
    
    # ## ------ end exec.  -----
    _exportResult(results, DataTaurus.export_list_results)
    print("   ** generate File 1st convergence in:", DataTaurus.export_list_results)
    if results_voidStep:
        _exportResult(results_voidStep, voidDD_path)
        print("   ** generate File VoidStep in:", voidDD_path)


def mainLinuxSweepingPES(z, n, interaction, b_min=-0.1, b_max=0.1, N_max=50,
                         invert=False, voidDD_path=None):
    """ 
        Old process that sets an even single-evaluated step over b range
    voidDD_path is the equivalent of the DataTaurus.export_list_results for the 
    output of the final calculation
    """
    #%% Executing the process, run the list of isotopes
    #
    EXE_B2 = True
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    if voidDD_path and os.path.exists(voidDD_path):
        os.remove(voidDD_path)
    
    N_max += 1 
    b20_base = b_min if not invert else b_max
    b20_lim0 = b20_base
    b20_lim  = b_max if not invert else b_min 
    results          = [None] * N_max
    results_voidStep = [None] * N_max
    
    print(HEAD)
    reuse_wf = False
    if reuse_wf: print("\n EEEEEH! REUSING THE WAVEFUNCTION ****** \n\n"*3)
    # create a spherical/unbounded seed to proceed
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    constr_b20 = "1  {:+5.4f}".format(b20_base)
    constr_b20 = "0   {:5.4f}".format(b20_base)
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.z : z,     Template.n : n,
        Template.seed : 3 if not reuse_wf else 1,
        Template.grad_type : 1,
        Template.grad_tol  : 0.001,
        Template.eta_grad : 0.08,
        Template.iterations : 500 if not reuse_wf else 0,
        Template.mu_grad  : 0.15,
        Template.b20  : constr_b20,
    }
    RDIM_0, OMEG_0 = 12, 14
    _setDDTermInput(True, rdim=RDIM_0, omega=OMEG_0)
    
    print(" * first convergence (seed={}), interact={}"
            .format(kwargs[Template.seed], interaction))
    for iter_ in range(1, 4):
        res_0 = _executeProgram(kwargs, output_filename, 0.0)
        print(kwargs)
        if (res_0 != None) and res_0.properly_finished:
            break
        else:
            if iter_ == 3:
                print("[ERROR], after 4 tries the calculation STOP for", z, n)
                return
        kwargs[Template.eta_grad]   -= 0.007 * iter_
        kwargs[Template.eta_grad]    = max(kwargs[Template.eta_grad], 0.001)
        kwargs[Template.iterations] += 150 * (iter_ + 1)
        print(" [WARNING] 1st step non converged, next try:", iter_, kwargs[Template.eta_grad])
    _e = subprocess.call('cp final_wf.bin initial_Spheric.bin', shell=True)
    print("   ... done.")
    
    if EXE_B2 == True:
        b20_base = res_0.b20_isoscalar
    else: # TODO: fix for other deformations, (modify the input template)
        b20_base = res_0.b32_isoscalar
    
    if abs(b20_base - b20_lim0) > 0.01:
        print(" [WARNING] Modifying the b20 BASE[{:5.3f}] to {:5.3f}"
              .format(b20_lim0, b20_base))
    
    N_2 = int(N_max * abs((b20_lim - b20_base) / (b20_lim - b20_lim0))) + 1
    # ### N_2 = N_max if b20_lim0 = b20_base and the arrays remain same.
    deform_array = [
        list(np.linspace(b20_base, b20_lim, num=N_2, endpoint=True)),
        list(np.linspace(b20_lim0, b20_lim, num=N_max, endpoint=True))
    ]
    
    ## WARNING! compromising start point
    print(" [WARNING]! compromising start point")
    _printInput4Calculation(dq1=deform_array[0][1]-deform_array[0][0],
                            dq2=deform_array[1][1]-deform_array[1][0], 
                            Nmax=N_max, N2 = N_2,
                            b_LIMS=(b20_lim0, b20_base, b20_lim), 
                            voidDD_path=voidDD_path, 
                            deform_array = deform_array,
                            rdim=RDIM_0, omega=OMEG_0, **kwargs)  
    print(HEAD)
    deform_read = [list() for _ in range(N_max)] 
    for reverse, N in ((0, N_2), (1, N_max)):
        print('\n==== REVERSE READING [', bool(reverse), '] ==================\n')
        for i in range(N):
            i2 = i
            if reverse:
               i2 = - i - 1
            
            b20 = deform_array[reverse][i2]
            
            ## deactivate DD
            kwargs = {
                Template.interaction : interaction, 
                Template.com : 1,   Template.read_red_hamil : 1,
                Template.z : z,     Template.n : n,
                Template.seed : 1,
                Template.grad_type  : 0,
                Template.grad_tol   : 0.005,
                Template.eta_grad   : 0.015,
                Template.iterations : 500,
                Template.mu_grad    : 0.4,
                Template.b20  : "1 {:+5.4f}".format(b20),
            }
            #res = _convergence_loop(kwargs, output_filename, b20, setDD=False)
            _setDDTermInput(True, rdim=10, omega=14)
            res = _executeProgram(kwargs, output_filename, b20,
                                  save_final_wf=True)
            if res == None:
                continue # dont save empty result
            
            if reverse:
                if results[i2] != None:
                    
                    if results[i2].E_HFB < res.E_HFB:
                        # deform_read[i2].append("rev: b= {:8.5f} [{}] REJ".format(b20, i2))
                        continue # don't save, new energy is bigger
                    # else:  # REMOVE
                    #     deform_read[i2].append("rev: b= {:8.5f} [{}] ACP".format(b20, i2))
                else:
                    results[i2] = res
                    # deform_read[i2].append("rev: b= {:8.5f} [{}] NEW".format(b20, i2))
            elif not reverse:
                ind_dir = N_max - N_2 + i2
                results[ind_dir] = res
                # deform_read[ind_dir].append("dir: b= {:8.5f} [{}]".format(b20, ind_dir))
            
            # includes direct result, reverse valid over None, and E' < E
            # intermediate print
            # print("results ARRAY:\n[")
            # for i, val in enumerate(results):
            #     if val:
            #         print(i, ":", val.b20_isoscalar)
            #     else:
            #         print(i, ":", val)
            # print("]\ndeform_read ARRAY:\n[")
            # for i, val in enumerate(deform_read):
            #     print(i, ":", val)
            # print("]")
            _exportResult(results, DataTaurus.export_list_results)
            
            ## SECOND PROCESS --------------------------------
            if voidDD_path != None:
                
                ##do a void step to activate DD with no rearrangement
                _setDDTermInput(True, rdim=8, omega=8)
                kwargs = {
                    Template.interaction : interaction, 
                    Template.com : 1,   Template.read_red_hamil : 1,
                    Template.z : z,     Template.n : n,
                    Template.seed : 1,
                    Template.grad_type : 0,
                    Template.grad_tol  : 0.05,
                    Template.eta_grad : 0.002,
                    Template.iterations : 30,
                    Template.mu_grad  : 0.01,
                    Template.b20  : "1 {:5.4}".format(b20),
                }
                print("2:", output_filename, b20)
                _setDDTermInput(True, rdim=8, omega=10)
                res2 = _executeProgram(kwargs, output_filename+'_VS_', b20,
                                       save_final_wf=True)
                print()
                
                if res2 == None:
                    continue # dont save empty result
                
                if reverse:
                    if results_voidStep[i2] != None:
                        if results_voidStep[i2].E_HFB < res2.E_HFB:
                            continue # don't save, new energy is bigger 
                        else:
                            results_voidStep[i2] = res2
                else:
                    ind_dir = N_max - N_2 + i2
                    results_voidStep[ind_dir] = res2 
            
            # intermediate print
            if voidDD_path != None:
                _exportResult(results_voidStep, voidDD_path) 
    
    # ## ------ end exec.  -----
    _exportResult(results, DataTaurus.export_list_results)
    print("   ** generate File 1st convergence in:", DataTaurus.export_list_results)
    if voidDD_path != None:
        _exportResult(results_voidStep, voidDD_path)
        print("   ** generate File VoidStep in:", voidDD_path)
    
    
    # print("deform_read ARRAY:")
    # for i, val in enumerate(deform_read):
    #     print(i, ":", val)

def _energyDiffRejectionCriteria(curr_energ,  old_energ, old_e_diff, 
                                       tol_factor=2.0):
    new_e_diff = curr_energ - old_energ
    # change in direction of the derivative, reject if difference is > 25%
    if new_e_diff * old_e_diff < 0: 
        return abs(new_e_diff) > 2.0 * abs(old_e_diff)
    # reject if new difference is tol_factor greater than the last one.
    return abs(new_e_diff) > tol_factor * abs(old_e_diff)

def mainLinuxSecurePES(z, n, interaction, b_min=-0.1, b_max=0.1, N_base=50, 
                       b20_base=None, voidDD_path=None):
    """ 
        Process that evaluates the deformation limits fitting the q20 to not 
    phase breaking, q20 is reduced up to 2^3 of the evenly spaced step.
        The criteria to continue the iteration is the HFB/Kin energy jump 
    for the new step (pair omitted since pair=0.0 is common)
    
        !! Note the increment of N_base will just increase the precision,
    dq_base will be progressively smaller (if it's stuck in a point you will
    need to increase the factor of the N_MAX limit)
    """
    #%% Executing the process, run the list of isotopes
    #
    
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    if voidDD_path and os.path.exists(voidDD_path):
        os.remove(voidDD_path)
    
    results = []
    results_voidStep = []
    
    ## definitions for the iteration
    dq_base  = max((b_max - b_min) / N_base, 0.008)
    b20_base = 0.0000 if not b20_base else b20_base
    ener_base = None
    N_MAX = 15 * N_base
    dqDivisionMax = 3
    
    
    print(HEAD)
    # create a spherical/unbounded seed to proceed
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    # kwargs = {
    #     Template.interaction : interaction, 
    #     Template.com : 1,   Template.read_red_hamil : 0,
    #     Template.z : z,     Template.n : n,
    #     Template.seed : 3,
    #     Template.grad_type : 1,
    #     Template.grad_tol  : 0.007,
    #     Template.eta_grad : 0.010,
    #     Template.iterations : 1000,
    #     Template.mu_grad  : 0.25,
    #     Template.b20  : "0   0.000",
    # }
    # _setDDTermInput(False, rdim=1, omega=1)
    # res_0 = _executeProgram(kwargs, output_filename, b20_base, save_final_wf=True)
    ## %%%%%%%% ()
    
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0,
        Template.z : z,     Template.n : n,
        Template.seed : 3,
        Template.grad_type : 1,
        Template.grad_tol  : 0.007,
        Template.eta_grad : 0.015,
        Template.iterations : 400,
        Template.mu_grad  : 0.15,
        Template.b20  : "1 {:+5.4f}".format(b20_base),
    }
    print(" * first convergence (seed{}), interact={}"
            .format(kwargs[Template.seed], interaction))
    ## deactivate DD for VOID STEP
    RDIM_0, OMEG_0 = 10, 12
    _setDDTermInput(True, rdim=RDIM_0, omega=OMEG_0)
    # _setDDTermInput(False, rdim=1, omega=1) (only for voidDD step)
    
    _printInput4Calculation(dq=dq_base, b_LIMS=(b_min,b20_base,b_max), N_MAX=N_MAX,
                            voidDD_path=voidDD_path, rdim=RDIM_0, omega=OMEG_0, **kwargs)
    
    for iter_ in range(1, 4):
        res_0 = _executeProgram(kwargs, output_filename, b20_base)
        if (res_0 != None) and res_0.properly_finished:
            break
        else:
            if iter_ == 3:
                print("[ERROR], after 4 tries the calculation STOP for", z, n)
                return
        kwargs[Template.eta_grad]   -= 0.007 * iter_
        kwargs[Template.eta_grad]    = max(kwargs[Template.eta_grad], 0.001)
        kwargs[Template.iterations] += 150 * iter_
        print(" [WARNING] 1st step non converged, next eta:", iter_, kwargs[Template.eta_grad])
    # First convergence done
    ener_base = float(res_0.E_HFB)
    _e = subprocess.call('cp final_wf.bin initial_Spheric.bin', shell=True)
    print("   ... done.")
    results.append(res_0)
    
    ## WARNING! compromising start point
    b20_base = float(res_0.b20_isoscalar)
    # b20_base = float(res_0.b30_isoscalar)
    print(" WARNING! compromising start point b20=", b20_base)
    
    # ###
    for prolate, b_lim in enumerate((b_min, b_max)):    #prolate = 1          
        # copy the first function.
        _e = subprocess.call('cp initial_Spheric.bin initial_wf.bin', shell=True)
        
        b20_i  = b20_base
        energ  = ener_base
        curr_energ = 10.0
        e_diff = 10.0 #  
        i = 0
        div = 0
        print(" RUNNING DEFORM. [",prolate,"] up to:", b_lim, N_MAX, dq_base)
        
        _whileCond = b20_i < b_lim if (prolate==1) else b20_i > b_lim
        while _whileCond and (i < N_MAX):
            b20 = b20_i - ((-1)**(prolate)*(dq_base / (2**div)))
                        
            # execute but do not save the final function
            kwargs = {
                Template.interaction : interaction, 
                Template.com : 1,   Template.read_red_hamil : 1,
                Template.z : z,     Template.n : n,
                Template.seed : 1,
                Template.grad_type : 0,
                Template.grad_tol  : 0.005,
                Template.eta_grad : 0.01,
                Template.iterations : 500,
                Template.mu_grad  : 0.3,
                Template.b20  : "1 {:5.4f}".format(b20),
            }
            if div > 3: 
                # avoid the steps of eta adjustment for big dq_,
                # Usually, splitting the b20 can converge the result and avoid 
                # us the 3 extensive steps of convergence
                res = _convergence_loop(kwargs, output_filename, b20,
                                        save_final_wf=False, setDD=False)
            else: 
                # _setDDTermInput(True, rdim=4, omega=3)
                # if voidDD_path:
                _setDDTermInput(True, RDIM_0, OMEG_0)
                res = _executeProgram(kwargs, output_filename, b20, 
                                      print_result=True, save_final_wf=False, 
                                      force_converg = True)
            
            ## Case 1: the program broke and the result is NULL
            if res == None:
                i += 1 
                if div < dqDivisionMax:
                    # reject(increase division)
                    div += 1
                    print("  * reducing b20 increment(1): [{}] Ei{:9.2f} - Eim1{:9.2f} ={:8.5f} > {:8.5f}"
                          .format(div, curr_energ, energ, curr_energ - energ, e_diff))
                    continue
                else:
                    # accept and continue (DONT copy final function)
                    # increase the step for valid or deformation precision overflow 
                    div = max(0, div - 1) ## smoothly recover the dq
                    e_diff = curr_energ - energ
                    energ = curr_energ
                    b20_i = b20
                    print("  * Failed but continue: DIV{} DIFF{:10.4f} ENER{:10.4f} B{:5.3f}"
                          .format(div, e_diff, energ, b20_i))
                    continue # cannot evaluate next Step or save results
            
            ## Case 2: the program did'nt broke and the result has values
            # take the E_HFB energy and compare the previous (acceptance criteria)
            curr_energ = float(res.E_HFB)
            i += 1
            if ((div < dqDivisionMax)
                and (_energyDiffRejectionCriteria(curr_energ, energ, e_diff, 
                                                  tol_factor=2.5)
                     or (not res.properly_finished))):
                # reject(increase division)
                div += 1
                print("  * reducing b20 increment(2)[i{}]: [{}] Ei{:9.2f} - Eim1{:9.2f} ={:8.5f} >  ({:8.5f}, {:8.5f})"
                      .format(i,div, curr_energ, energ, curr_energ-energ, e_diff,
                              3.0*e_diff, 2.0*e_diff))
                continue
            else:
                print("  * [OK] step accepted DIV:{} CE{:10.4} C.DIFF:{:10.4}"
                      .format(div, curr_energ, curr_energ - energ))
                # accept and continue (copy final function)
                _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
                # increase the step for valid or deformation precision overflow 
                div =  max(0, div - 1) ## smoothly recover the dq
                e_diff = curr_energ - energ
                energ = curr_energ
                b20_i = b20
                print("  * [OK] WF directly copied  [i{}]: DIV:{} DIFF{:10.4f} ENER{:10.4f} B{:5.3f}"
                      .format(i, div, e_diff, energ, b20_i))
            
            if prolate == 0:
                results.insert(0, res) #grow in order [-.5, -.4, ..., .0,..., +.4, +.5] 
            else:
                results.append(res)   
            
            ## SECOND PROCESS --------------------------------
            if voidDD_path != None:  # VOID STEP
                ##do a void step to activate DD with no rearrangement
                # _setDDTermInput(True, rdim=10, omega=10)
                ## increase of the convergece precision
                _setDDTermInput(True, rdim=8, omega=10)
                kwargs = {
                    Template.interaction : interaction, 
                    Template.com : 1,   Template.read_red_hamil : 1,
                    Template.z : z,     Template.n : n,
                    Template.seed : 1,
                    Template.grad_type : 0,
                    Template.grad_tol  : 0.03,
                    Template.eta_grad : 0.0031,
                    Template.iterations : 100,  # 1000# change 0 for void step
                    Template.mu_grad  : 0.051,
                    Template.b20  : "1 {:5.4}".format(b20),
                }
                res2 = _executeProgram(kwargs, output_filename+'_VS_', b20,
                                       save_final_wf=False)
            
                print("-------------------------------------------------------------------------------")
                print()
                
                if res2 == None:
                    continue # dont save empty result
                if prolate == 0:
                    results_voidStep.insert(0, res2)
                else:
                    results_voidStep.append(res2)   
            
            # intermediate print
            _exportResult(results, DataTaurus.export_list_results)
            if voidDD_path:
                _exportResult(results_voidStep, voidDD_path) 
    
    # ## ------ end exec.  -----
    _exportResult(results, DataTaurus.export_list_results)
    print("   ** generate File 1st convergence in:", DataTaurus.export_list_results)
    if voidDD_path != None:
        _exportResult(results_voidStep, voidDD_path)
        print("   ** generate File VoidStep in:", voidDD_path)


def _exportResult(results, path_):
    data = []
    for res in results:
        if res:
            line = res.getAttributesDictLike
            data.append(line+'\n')
    
    with open(path_, 'w+') as f:
        f.writelines(data)

def plot_propertiesAxialTaurus(properties, results_taur, results_axial):
    
    for attr_ in properties:
        ## plot energies
        x_tau, y_tau, xb_tau = [], [], []
        x_ax,  y_ax  = [], []
        
        try:
            minima = min(min([getattr(r, attr_) for r in results_taur]),
                         min([getattr(r, attr_) for r in results_axial]))
        except BaseException:
            minima = 0
        
        for r in results_taur:
            q_ax = r.q20_isoscalar * (2*np.pi / ((5*np.pi)**.5))
            xb_tau.append(r.b20_isoscalar)
            # q_ax = r.b32_isoscalar
            # if q_ax < -30:
            #     continue
            x_tau.append(q_ax)
            
            # x_tau.append(r.b20_isoscalar)
            if attr_ == 'kin':
                y_tau.append(getattr(r, attr_))
            else:
                y_tau.append(getattr(r, attr_) - minima*0)
        
        for r in results_axial:
            # if r.q20_isoscalar < -30:
            #     continue
            # x_ax.append(r.beta_isoscalar)#_isoscalar)#q20_isoscalar)
            x_ax.append(r.q20_isoscalar)
            
            if attr_ == 'hf':
                y_ax.append(getattr(r, attr_) - getattr(r, 'kin'))
                print(y_ax)
            else:
                y_ax.append(getattr(r, attr_, 0.0) - minima*0)
        
        if attr_ == 'properly_finshed':
            y_ax = [1 if p == 'True' else 0 for p in y_ax]
        elif attr_ == 'hf':
            y_ax = [y - minima for y in y_ax]
        
        fig, ax = plt.subplots()
        # ax2 = ax.twiny()
        ax.set_xlabel(r"$Q_{20} [fm^2]$")
        
        # ax.set_ylabel()
        
        # ax.set_xlabel(r"$\beta_{\ 32}$")
        
        ax.plot(x_ax,  y_ax,  'o-b', label="HFB axial", alpha=0.2)
        ax.plot(x_tau, y_tau, '.-r', label="Taurus")
        
        ## double X axis for beta
        ax2 = ax.twiny()
        ax2.set_xlabel(r"$\beta_{20}$")
        ax2.set_xlim(ax.get_xlim())
        new_ticks = list(ax.get_xticks())[1:-1]
        ax2.set_xticks(new_ticks)
        C2_const =4*np.pi / (3*(1.2**2)* (A**(5/3)) * 1.5853309)
        new_tick_loc = [round(C2_const * qax,2) for qax in new_ticks]
        ax2.set_xticklabels(new_tick_loc)
        plt.grid()
        
        
        plt.title(attr_+" - min()  [Z:{} N:{}]  ".format(z, n)+ tail+
                  "\n"+"min({}) = {:8.3f}".format(attr_, minima))
        # plt.title("min({}) = {:8.3f}".format(attr_, minima), fontsize=10)
        
        # plt.title(r"$E_{HFB}\ [MeV] \quad {}^{22}Mg\ using\ D1S\  N_{max}=3 $")
        # plt.title(r"$\Delta\ (pairing\ E.)\ [MeV] \quad {}^{22}Mg\ using\ D1S\  N_{max}=3 $")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    
        # print the pdf images and the Latex-figure- Template
        isot_name = '{}{}'.format(n+z, nuclei[z])
        title_fig = 'Figure_'+isot_name+'_'+attr_+'_'+tail+'.pdf'
        # plt.savefig(title_fig)
    
    latex_fig = "\\subsubsection{"+isot_name+", Figure \\ref{fig:"+isot_name+"_"+tail+"_MZ2_FullnoLs}}\n" +\
"\\begin{figure}\n"+\
"    \\centering\n"+\
"    \\subfloat {{\\includegraphics[width=100mm]{SPSD_NeFullD1S/Figure_"+isot_name+"_E_HFB_"+tail+".pdf} }}\n"+\
"    \\qquad\n"+\
"    \\subfloat {{\\includegraphics[width=100mm]{SPSD_NeFullD1S/Figure_"+isot_name+"_hf_"+tail+".pdf} }}\n"+\
"    \\qquad\n"+\
"    \\subfloat[\\centering Kinetic energy (MeV)]\n"+\
"    {{\\includegraphics[width=85mm]{SPSD_NeFullD1S/Figure_"+isot_name+"_kin_"+tail+".pdf} }}\n"+\
"    \\subfloat[\\centering Pairing energy (MeV)]\n"+\
"    {{\\includegraphics[width=85mm]{SPSD_NeFullD1S/Figure_"+isot_name+"_pair_"+tail+".pdf} }}\n"+\
"    \\caption{Calculation for "+isot_name+" on the SPSD shell space. b length = "+str(b_lenght)+" fm}\n"+\
"    \\label{fig:"+isot_name+"_D1S_MZ2}\n"+\
"\\end{figure}\n"
    print()
    print(latex_fig)
    print()



nuclei = {2:'He', 4:'Be', 6:'C', 8:'O', 10:'Ne', 12:'Mg', 14:'Si', 16:'S', 18:'Ar', 20:'Ca',
          34:'Se', 36:'Kr', 38:'Sr'}

nucleus = []
for z in range(12,15, 2):
    for n in range(max(6, z-2), 17, 2):
        nucleus.append((z, n))

if __name__ == '__main__':
    
    b_lenght = 1.8
    z = 12 #6 #
    n = 10 #8 #
    
    tail = ''
    # tail = '1GCOM0'
    # tail = 'B1'
    # tail = 'B1COM0' 
    # tail   = 'D1Scom0'
    # tail   = 'D1SnoR'
    # tail = 'D1S_Fixed'
    tail = 'D1S'
    # tail = 'only_DD'
    # tail = 'D1S_voidDD' #'D1SnoR'
    
    # interaction = 'D1S_t0_SPSDPF' #'D1S_t0_SPSDPF' # 'B1_SPSD' #'0s0p1s0d'  #
    interaction = 'D1S_noLSt0_MZ2'
    # interaction = 'D1S_t0_SPSD'
         
    if not os.getcwd().startswith('C:'):    #
        assert "voidDD" not in tail, "[ERROR] [STOP] 'voidDD' in tail for exporting name!"
        print()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('%%%   Running PES with Taurus_VAP, hamilonian:', interaction)
        print('%%%     !!! CHECK, CHECK MZ:', interaction.upper())
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print("GLOBAL START:", datetime.now().time())
        print()
        
        for z, n in nucleus:
            print("Start mainLinux: ", datetime.now().time())
            print("PES for Z={} N={}".format(z,n))
            DataTaurus.export_list_results = "export_PESz{}n{}Taurus{}.txt".format(z,n, tail)
            # voidDD_path = "export_PESz{}n{}Taurus{}_voidDD.txt".format(z,n, tail)
            voidDD_path = "export_PESz{}n{}Taurus{}.txt".format(z,n, tail)
            #recoverResults(z, n)
            
            # mainLinuxDDFixed(z, n, interaction)
            # mainLinuxEvenlyDeform(z, n, interaction, -0.28,0.28, 60,voidDD_path)
            # mainLinuxSecurePES(z, n, interaction, -0.6, 0.6, 90, 
            #                    voidDD_path=None, b20_base=0.6)
            
            mainLinuxSweepingPES(z, n, interaction, -0.6, 0.6, 90, voidDD_path=None)
            
            print("End mainLinux: ", datetime.now().time())
    else:
        z = 12
        nucleus = [(z, 8), (z, 10), 
                   (z, 12), (z, 14), (z, 16)]#,
        
        for z, n in nucleus:
            #%% process in windows
            results_taur = []
            results_axial = []
            
            # z = 16 #6 #
            # n = 16 # 
            
            # z = 34 #6 #
            # n = 36 #8 #
            A = n + z
            #from exe_isotopeChain_axial import DataAxial
            PESfolder = 'BU_results_old/BU_voidDD_withConvergenceSaved/'
            PESfolder = 'BU_results_old/SPSD/'
            PESfolder = 'BU_results_old/SPSD_Mg_noLS/'
            PESfolder = 'BU_results_old/SPSD_NefullD1S/'
            PESfolder = 'BU_results_old/SPSD_FullD1Sb177/'
            # PESfolder = 'BU_results_old/SPSD_noLSb18/'
            
            # PESfolder = 'BU_results_old/SPSDPF_Nefull_noLS/'
            # PESfolder = 'BU_results_old/SPSDPF_Ne_noLS_voidStep/' 
            # PESfolder = 'BU_results_old/SPSD_onlyDD/'
            # PESfolder = 'BU_results_old/SPSDPF_NefullD1S/'
            
            PESfolder = 'BU_q20results/MZ2_test_Nov22/'
            # PESfolder = 'BU_q20results/MZ3_test_Nov22/'
            # PESfolder = 'BU_q20results/MZ2_MgSitest_Nov28/'
            # PESfolder = 'PN_mixingD1STests/'
            # PESfolder = 'BU_q20results/SPSDPF/FullD1S/'
            # PESfolder = 'BU_q20results/MZmax5/'
            # PESfolder = 'BU_q20results/SPSDPF/Q32FullD1S/'
            # PESfolder = 'BU_q20results/SPSD/noLS_FullD1S/'
            # PESfolder = 'BU_q20results/SPSD/FullD1S/'
            
            # tail = 'D1S_s3'
            
            DataTaurus.export_list_results = "export_PESz{}n{}Taurus{}.txt".format(z,n, tail)
            import_file_Tau = PESfolder + DataTaurus.export_list_results        
            import_file_Axi = PESfolder + 'export_PESz{}n{}Axial{}.txt'.format(z,n,tail)
            
            ## get results from export "csv"
            print(import_file_Tau,'\n',import_file_Axi)
            if os.path.exists(import_file_Tau):
                with open(import_file_Tau, 'r') as f:
                    data = f.readlines()
                    for line in data:
                        res = DataTaurus(None, None, None, True)
                        res.setDataFromCSVLine(line)
                        results_taur.append(res)
            else:
                print("[ERROR] couldn't find ", import_file_Tau)
            if os.path.exists(import_file_Axi):
                with open(import_file_Axi, 'r') as f:
                    data = f.readlines()
                    for line in data:
                        res = DataAxial(None, None, None, True)
                        res.setDataFromCSVLine(line)
                        results_axial.append(res)
            else:
                print("[ERROR] couldn't find ", import_file_Axi)
            
            properties2show = (
                'E_HFB', 
                # 'kin',
                # 'hf',
                # 'b11_p',
                # 'b11_n',
                # 'Q_10_p',
                # 'Q_10_n',
                # 'var_p', 
                'pair',#, 'properly_finished'
                # 'pair_pp',
                # 'pair_nn',
                # 'Jx_var',
                # 'Jz',
                # 'r_isoscalar',
                # 'time_per_iter',
                # 'iter_max',
                # 'iter_time_seconds',
            )
            
            plot_propertiesAxialTaurus(properties2show, results_taur, results_axial)
        
    
#%%


    import numpy as np
    def B(ZA):
        return np.sqrt(41.4/((45.0*ZA**(-1.0/3.0))-(25.0*ZA**(-2.0/3.0))))
    
    def B_hbarO(hbarO):
        return (((197.327053**2) / (938.91875434 * hbarO)))**0.5
    
    def hbarO(b):
        return ((197.327053**2) / 938.91875434) /(b**2)

    N_max = 13
    
    LSTATES = 'spdfghijklmnopqrst'
    GLOBAL_N = 0
    # for N in range(N_max):
    #     par    = N % 2
    #     Lgroup = []
    #     for l in range(N+1):
    #         if l % 2 != par: continue
    #         Lgroup.append(l)
            
    #     sh_states = []
    #     print("***  N = ",N,  " "+ "***"*6)
    #     total_deg = 0
    #     for n in range(0, N//2 + 1):
            
    #         for l in Lgroup:
    #             if 2*n + l != N: continue
                
    #             for s in (1, -1):
    #                 j = 2*l + s
    #                 if j < 0: continue
    #                 ant = 10000*n + 100*l + j
    #                 sub_deg = j + 1
    #                 str_ = "{}{}{:2}/2 [{:5}]   deg={:2}"\
    #                     .format(n,LSTATES[l],j,ant,sub_deg)
    #                 sh_states.append(str)
                    
    #                 if s == -1 or l == 0:
    #                     deg = (2*l + 1) * 2
    #                     str_ += "  * L_deg({})= {}".format(LSTATES[l].capitalize(), deg)
    #                 print(str_)
                
    #             total_deg += deg
        
    #     GLOBAL_N += total_deg
    #     print("*** SHELL DEGENERATION =", total_deg, " / global =", GLOBAL_N)
    #     print()
    
    
