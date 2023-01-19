"""
Created on Fri Mar  4 19:28:46 2022

@author: Miguel
"""
from _legacy.exe_isotopeChain_taurus import DataTaurus

class Template:
    com = 'com'
    z   = 'z'
    a   = 'a'
    seed    = 'seed'
    b20     = 'b20'
    varN2   = 'varN2'
    iterartions = 'iters'
    hamil = 'interaction'

TEMPLATE = """NUCLEUS    {a:03} XX   Z= {z:03}     >>> HFB OPT <<< {com} COULECH 2
EPSG 0.000001 MAXITER {iters:05}    >>> OUTPUT  <<< 0 **** TSTG 0
ETMAX 0.7501 ETMIN 0.0351 DMAX 0.90 DMIN 0.70 TSHH 399.0
GOGNY FORCE        {interaction}    *** 0=D1S 1=D1 2=D1' 3(t3=0)
INPUT W.F.         {seed}    *** 0,1=WF FROM UNIT 10 (1 kicks), 2=NEW Function
OSCILLATOR LENGHT  0    *** 0               BP 1.7510000 BZ 1.7510000
          >>>>>>>>>> C O N S T R A I N T S <<<<<<<<<<<
C.O.M.     1 1   0.00000000D+00
{b20}{varN2}          >>>>>>>>>> E N D <<<<<<<<<<<<<<<<<<<<<<<<<<<  """
# BP 1.7719772 BZ 1.7719772
# BP 1.7185258 BZ 1.7185258 (A=25)
temp_noGrad = """NUCLEUS    {a:03} He   Z= {z:03}     >>> HFB OPT <<< {com} COULECH 2
EPSG 0.000001 MAXITER {iters:05}    >>> OUTPUT  <<< 0 **** TSTG 0
ETMAX 0.0001 ETMIN 0.0001 DMAX 0.01 DMIN 0.01 TSHH 000.1
GOGNY FORCE        {interaction}    *** 0=D1S 1=D1 2=D1' 3(t3=0)
INPUT W.F.         {seed}    *** 0,1=WF FROM UNIT 10 (1 kicks), 2=NEW Function
OSCILLATOR LENGHT  0    *** 0               BP 2.0402454 BZ 2.0402454
          >>>>>>>>>> C O N S T R A I N T S <<<<<<<<<<<
C.O.M.     1 1   0.00000000D+00
{b20}{varN2}          >>>>>>>>>> E N D <<<<<<<<<<<<<<<<<<<<<<<<<<<  """


q10_constr_template = "QL    1    {:1} {:1}   {:10.8f}D+00\n"
q20_constr_template = "QL    2    {:1} {:1}   {:10.8f}D+00\n"
b20_constr_template = "BL    2    {:1} {:1}   {:10.8f}D+00\n"
b30_constr_template = "BL    3    {:1} {:1}   {:10.8f}D+00\n"
DN2_constr_template = "DN**2      {:1} {:1}   {:10.8f}D+00\n"
DJX2_constr_template= "DJX**2     {:1} {:1}   {:10.8f}D+00\n"
MSR2_constr_template= "<R**2>     {:1} {:1}   {:10.8f}D+00\n"


com_template   = "CM1 {} CM2 {}"


from collections import OrderedDict
import os
import shutil
import subprocess

from _legacy.exe_isotopeChain_axial import DataAxial
import math as mth
import matplotlib.pyplot as plt
import numpy as np


HAMIL_AXIAL_PROGRAM = 'HFBaxialMZ3'

nucleus = [
#    Z  N
    # (2, 2), 
    # (2, 4),
    # (4, 4), 
    # (4, 6),
    # # (6, 6), 
    # (6, 8),
    
    # (8, 4),
    # (8, 6),
    # (8, 8),
    # (8, 10),
    # (8, 12),
    #
    (10, 6),
    (10, 8),
    (10, 10),
    (10, 12),
    (10, 14),
    (10, 16)
    #

    # (12, 8),
    # (12, 10),
    # (12, 12),
    # (12, 14),
    # (12, 16),
    
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
    
    # (36, 34),
    # (34, 36),
    # (38, 40),
    # (40, 38),
]
#nucleus = [(8, n) for n in range(6, 15, 2)]

## put here value in axial (divide by 3/5 to fix with taurus q20)
repeat = {
    # # (2, 2), 
    # (2, 4) : 0.1 / 0.6,
    # (4, 4) : 0.4 / 0.6, 
    # # (4, 6) : -0.4 / 0.6,
    # # (6, 6) : -0.4 / 0.6, 
    # # (6, 8) : -0.1 / 0.6,
    # # (8, 8),
    # (10, 8) : 0.0,
    # # (10, 10): +0.2 / 0.6,
    # (12, 10): +0.3 / 0.6,
    # (14, 12) : 0.23 / 0.6,

    # #(6, 6) : -0.4  / 0.6,
}

def _executeProgram(params, output_filename, q20_const, 
                    print_result=True, save_final_wf=True, force_converg=False,
                    noGradient=False):
    """
    In NOT save_final_wf, the initial wf previous the calculation is restored
    """
    res = None
    if params[Template.seed] == 1: print("[WARNING] seed 1 in Axial kicks wf!")
    
    try:
        status_fin = ''        
        
        text = TEMPLATE.format(**params)
        if noGradient:
            text = temp_noGrad.format(**params)
            #print("\n no Grad\n{}".format(text),'\n')
        with open(DataAxial.INPUT_FILENAME, 'w+') as f:
            f.write(text)
        #_e = subprocess.call('cp fort.10 initial_fort.10', shell=True)
        
        _e = subprocess.call('./{} < {} > {}' # noReaHFBMZ2
                                    .format(HAMIL_AXIAL_PROGRAM,
                                            DataAxial.INPUT_FILENAME, 
                                            output_filename), 
                              shell=True)
        res = DataAxial(z, n, output_filename)
        
        # move shit to the folder
        str_q20 = str(int(1000*q20_const)).replace('-','_')
        folder_dest = os.getcwd()+'/'+DataAxial.BU_folder+'/'
        _e = subprocess.call('mv {} {}'.format(output_filename, 
                              folder_dest+output_filename
                              +'_Z{}N{}'.format(z,n)
                              +'_{}'.format(str_q20)),
                              shell=True)
        _e = subprocess.call('cp fort.11 '+folder_dest+
                             'seed_q{}_'.format(str_q20)+
                             '_Z{}N{}'.format(z,n)+'.11', 
                             shell=True)
        #_e = subprocess.call('cp fort.11 final_fort.11', shell=True)
        _e = subprocess.call('rm fort.38 fort.4* fort.5* fort.6*', shell=True)
        
        # refresh the initial function to the new deformation
        if save_final_wf and (res.properly_finished or (not force_converg)):
            _e = subprocess.call('rm fort.10', shell=True)
            _e = subprocess.call('cp fort.11 fort.10', shell=True)
            print("      *** exec. [OK] copied the final wf to the initial wf!")
        # else:
        #     _e = subprocess.call('cp initial_fort.10 fort.10', shell=True)
        
        status_fin = 'X' if not res.properly_finished else '.'
        if print_result:      
            print(" {:2} {:2}  ( {})    {:9.4f}  {:9.4f}  {:7.4f}  {:5.4f}={:6.2f}"
                  .format(z, n, status_fin, res.E_HFB, res.kin, res.pair, 
                          res.beta_isoscalar, res.q20_isoscalar))
    except Exception as e:
        print(" >> EXCEP >>>>>>>>>>  ")
        print(" >> current b20 =", q20_const)
        print(" > [",e.__class__.__name__,"]:", e, "<")
        if res and res.E_HFB == None and not res.properly_finished:
            print(" > the result is NULL (final_wf wasn't copied to initial_wf)")
            print("> RESULT <DataAxial>:\n",res,"\n END RESULT <")
        print(" << EXCEP <<<<<<<<<<  ")
        return None
    
    return res

def _energyDiffRejectionCriteria(curr_energ,  old_energ, old_e_diff, 
                                       tol_factor=2.0):
    new_e_diff = curr_energ - old_energ
    # change in direction of the derivative, reject if difference is > 25%
    if new_e_diff * old_e_diff < 0: 
        return abs(new_e_diff) > 1.5 * abs(old_e_diff)
    # reject if new difference is tol_factor greater than the last one.
    return abs(new_e_diff) > tol_factor * abs(old_e_diff)

def _set_deform_for_PES(res_0, b_min=-0.3, b_max=0.3, N = 20):
    """
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

def mainLinuxEvenlyDeform(z, n, b_min=-0.1, b_max=0.1, N=30, voidDD_path=None):
    """ 
        Old process that sets an even single-evaluated step over b range
    voidDD_path is the equivalent of the DataTaurus.export_list_results for the 
    output of the final calculation
    """
    #
    #%% Executing the process, run the list of isotopes
    #
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    A = n + z 
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataAxial.setUpFolderBackUp()
    if os.path.exists(DataAxial.export_list_results):
        os.remove(DataAxial.export_list_results)
    if voidDD_path and os.path.exists(voidDD_path):
        os.remove(voidDD_path)
    
    results = []
    results_voidStep = []
    
    print(HEAD)
    constr_N2, constr_DJ2, constr_MSR = '', '', ''
    constr = ''
    
    # create a spherical seed to proceed
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    # constr  = q20_constr_template.format(1,1, 0.0000)
    # constr = b20_constr_template.format(1,0, 0.0000)
    constr += b20_constr_template.format(1,1, b_max-0.01)
    kwargs = {
        Template.com : com_template.format(1,1),
        Template.z   : z,
        Template.a   : A,
        Template.seed : 2,
        Template.iterartions : 2000,
        Template.b20   : constr, #"",
        Template.hamil : 0,
        Template.varN2 : constr_N2 + constr_DJ2 + constr_MSR
    }   
    print(" * first convergence (seed2)")
    _ = _executeProgram(kwargs, output_filename, 0.0)
    res_0 = _executeProgram(kwargs, output_filename, 0.0)
    _e = subprocess.call('cp fort.11 initial_Spheric.11', shell=True)
    print("   ... done.")
    
    # ###
    deform_oblate, deform_prolate = _set_deform_for_PES(res_0, b_min,b_max, N)
    
    for i_deform, deform in enumerate((deform_oblate, deform_prolate)):
        
        # copy it.
        _e = subprocess.call('cp initial_Spheric.11 fort.10', shell=True)
        ## ----- execution ----
        for b20_const in deform:
            # create a spherical seed to proceed
            #q20_const *= 2 * np.pi / (np.sqrt(5 * np.pi))
            constr  = b20_constr_template.format(1,1, b20_const)
            
            kwargs = {
                Template.com : com_template.format(1,1),
                Template.z   : z,
                Template.a   : A,
                Template.seed : 0,
                Template.iterartions : 2000,
                Template.b20  : constr,
                Template.hamil : 0,
                Template.varN2: constr_N2 + constr_DJ2 + constr_MSR
            }
            
            res = _executeProgram(kwargs, output_filename, b20_const,
                                  print_result=False)
            if res == None:
                continue # dont save empty result
            if i_deform == 0:
                results.insert(0, res)
            else:
                results.append(res)   
            
            ## SECOND PROCESS --------------------------------
            if voidDD_path == None:
                continue
            
            # do a void step to activate DD with no rearrangement
            kwargs = {
                Template.com : com_template.format(1,1),
                Template.z   : z,
                Template.a   : A,
                Template.seed : 0,
                Template.iterartions : 0,
                Template.b20  : constr,
                Template.hamil : 0,
                Template.varN2: constr_N2 + constr_DJ2 + constr_MSR
            }
            res2 = _executeProgram(kwargs, output_filename+'_VS_', b20_const,
                                   save_final_wf=False, noGradient=True)
            if res2 == None:
                continue # dont save empty result
            if i_deform == 0: #grow in order [-.5, -.4, ..., .0,..., +.4, +.5] 
                results_voidStep.insert(0, res2)
            else:
                results_voidStep.append(res2)          
            
            # intermediate print
            _exportResult(results, DataAxial.export_list_results)
            if voidDD_path != None:
                _exportResult(results_voidStep, voidDD_path) 
            
    # ## ------ end exec.  -----
    _exportResult(results, DataAxial.export_list_results)
    print("   ** generate File 1st convergence in:", DataAxial.export_list_results)
    if results_voidStep:
        _exportResult(results_voidStep, voidDD_path)
        print("   ** generate File VoidStep in:", voidDD_path)


def mainLinuxSweepingPES(z, n, b_min=-0.1, b_max=0.1, N_max=30, 
                         invert=False, voidDD_path=None):
    """ 
        Process that starts from the limit of the PES, advances until the end
    and return from the limit point, if the surface fall along the way, in the
    backward process it will register the case E' < E and save
    """
    #
    #%% Executing the process, run the list of isotopes
    #
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    A = n + z 
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataAxial.setUpFolderBackUp()
    if os.path.exists(DataAxial.export_list_results):
        os.remove(DataAxial.export_list_results)
    if voidDD_path and os.path.exists(voidDD_path):
        os.remove(voidDD_path)
    
    N_max += 1 
    b20_base = b_min if not invert else b_max 
    b20_lim  = b_max if not invert else b_min 
    results          = [None] * N_max
    results_voidStep = [None] * N_max
    
    print(HEAD)
    constr_N2, constr_DJ2, constr_MSR = '', '', ''
    
    # create a spherical seed to proceed
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    # constr  = q10_constr_template.format(1,0, 0.0)
    # constr += q10_constr_template.format(0,1, 0.0)
    constr = b20_constr_template.format(1,1, b20_base)
    kwargs = {
        Template.com : com_template.format(1,1),
        Template.z   : z,
        Template.a   : A,
        Template.seed : 2,
        Template.iterartions : 2000,
        Template.b20   : constr, #"",
        Template.hamil : 8,
        Template.varN2 : constr_N2 + constr_DJ2 + constr_MSR
    }   
    print(" * first convergence (seed2)")
    _ = _executeProgram(kwargs, output_filename, 0.0)
    res_0 = _executeProgram(kwargs, output_filename, 0.0)
    _e = subprocess.call('cp fort.11 initial_Spheric.11', shell=True)
    print("   ... done.")
    
    # ###
    deform_array = list(np.linspace(b20_base, b20_lim, num=N_max, endpoint=True))
    
    for reverse in (0, 1):
        print('\n==== REVERSE READING [', bool(reverse), '] ==================\n')
        for i in range(N_max):
            i2 = i
            if reverse:
                i2 = - i - 1 
            
            b20_const = deform_array[i2]
            constr = b20_constr_template.format(1,1, b20_const)
            kwargs = {
                Template.com : com_template.format(1,1),
                Template.z   : z,
                Template.a   : A,
                Template.seed : 0,
                Template.iterartions : 2000,
                Template.b20  : constr,
                Template.hamil : 8,
                Template.varN2: constr_N2 + constr_DJ2 + constr_MSR
            }
            
            res = _executeProgram(kwargs, output_filename, b20_const,
                                  print_result=True)
            if res == None:
                continue # dont save empty result
            
            if reverse:
                if results[i2] != None:
                    if results[i2].E_HFB < res.E_HFB:
                        continue # don't save, new energy is bigger 
            results[i2] = res 
            # includes direct result, reverse valid over None, and E' < E
            
            # intermediate print
            _exportResult(results, DataAxial.export_list_results)
            ## SECOND PROCESS --------------------------------
            if voidDD_path != None:
            
                ## do a void step to activate DD with no rearrangement
                kwargs = {
                    Template.com : com_template.format(1,1),
                    Template.z   : z,
                    Template.a   : A,
                    Template.seed : 0,
                    Template.iterartions : 0,
                    Template.b20   : constr,
                    Template.hamil : 0,
                    Template.varN2 : constr_N2 + constr_DJ2 + constr_MSR
                }
                res2 = _executeProgram(kwargs, output_filename+'_VS_', b20_const,
                                       save_final_wf=False, noGradient=True)
                if res2 == None:
                    continue # dont save empty result
                if reverse:
                    if results_voidStep[i2] != None:
                        if results_voidStep[i2].E_HFB < res2.E_HFB:
                            continue # don't save, new energy is bigger 
                results_voidStep[i2] = res2     
            
            # intermediate print
            if voidDD_path != None:
                _exportResult(results_voidStep, voidDD_path) 
            
    # ## ------ end exec.  -----
    _exportResult(results, DataAxial.export_list_results)
    print("   ** generate File 1st convergence in:", DataAxial.export_list_results)
    if voidDD_path != None:
        _exportResult(results_voidStep, voidDD_path)
        print("   ** generate File VoidStep in:", voidDD_path)



def mainLinuxSecurePES(z, n, b_min=-0.1, b_max=0.1, N_base=50, b20_base=None, 
                       voidDD_path=None):
    """ 
        Process that evaluates the deformation limits fitting the q20 to not 
    phase breaking, q20 is reduced up to 2^3 of the evenly spaced step.
        The criteria to continue the iteration is the HFB/Kin energy jump 
    for the new step (pair omitted since pair=0.0 is common)
    
        !! Note the increment of N_base will just increase the precision,
    dq_base will be progressively smaller (if it's stuck in a point you will
    need to increase the factor of the N_MAX limit)
    """
    #
    #%% Executing the process, run the list of isotopes
    #
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    A = n + z 
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataAxial.setUpFolderBackUp()
    if os.path.exists(DataAxial.export_list_results):
        os.remove(DataAxial.export_list_results)
    if voidDD_path and os.path.exists(voidDD_path):
        os.remove(voidDD_path)
    
    results = []
    results_voidStep = []
    
    ## definitions for the iteration
    dq_base  = (b_max - b_min) / N_base
    b20_base = 0.0000 if not b20_base else b20_base
    ener_base = None
    N_MAX = 70 * N_base   # 7 * N_base
    dqDivisionMax = 6
    
    print(HEAD)
    constr_N2, constr_DJ2, constr_MSR = '', '', ''
    # create a spherical seed to proceed
    ## NOTE: spherical constraint fits better with the following constrained 
    ## process, avoid (if possible) the first seed to be the a deformed minimum
    # constr  = b20_constr_template.format(1,1, 0.0000)
    constr    = b20_constr_template.format(1,1, b20_base)
    # constr_N2 = DN2_constr_template.format(1,0,2.6925926)
    # constr_N2+= DN2_constr_template.format(0,1,2.7390982)
    kwargs = {
        Template.com : com_template.format(1,0),
        Template.z   : z,   Template.a   : A,
        Template.seed : 2,
        Template.iterartions : 2000,
        Template.b20   : constr, #"", # 
        Template.hamil : 8,
        Template.varN2 : constr_N2 + constr_DJ2 + constr_MSR
    }   
    print(" * first convergence (seed2)")
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
        kwargs[Template.iterations] += 150 * iter_
        print(" [WARNING] 1st step non converged, next eta:", iter_, kwargs[Template.eta_grad])
    # First convergence done
    ener_base = float(res_0.E_HFB)
    print("[Ener Base] =", ener_base)
    _e = subprocess.call('cp fort.11 initial_Spheric.11', shell=True)
    print("   ... done.")
    results.append(res_0)
    
    ## WARNING! compromising constraint
    b20_base = float(res_0.beta_isoscalar) 
    print(" WARNING! compromising start point b20=", b20_base)
    
    # ###
    for prolate, b_lim in enumerate((b_min, b_max)):    #prolate = 1    
        # copy the first function.
        _e = subprocess.call('cp initial_Spheric.11 fort.10', shell=True)
        
        b20_i  = b20_base
        energ  = ener_base
        curr_energ = ener_base
        e_diff = 10.0 #  
        i = 0
        div = 0
        print("runing deform[",prolate,"] up to:", b_lim, N_MAX)
        
        while (abs(b20_i) < abs(b_lim)) and i < N_MAX:
            b20 = b20_i - (((-1)**(prolate))*(dq_base / (2**div))) 
            
            # execute but do not save the final function
            constr  = b20_constr_template.format(1,1, b20)
            kwargs = {
                Template.com : com_template.format(1,0),
                Template.z   : z,
                Template.a   : A,
                Template.seed : 0,
                Template.iterartions : 2000,
                Template.b20  : constr,
                Template.hamil : 8, # 0,#
                Template.varN2: ""
            }
            
            res = _executeProgram(kwargs, output_filename, b20,
                                  print_result=True, save_final_wf=False,
                                  force_converg=True)

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
                                                  tol_factor= 2.0)
                     or (not res.properly_finished))):
                # reject(increase division)
                div += 1
                print("  * reducing b20 increment(2) [i{}]: [{}] Ei{:9.2f} - Eim1{:9.2f} ={:8.5f} > ({:8.5f}, {:8.5f})"
                      .format(i, div, curr_energ, energ, curr_energ - energ, 
                              3.0*e_diff, 1.5*e_diff))
                continue
            else:
                print("  * [OK] step accepted DIV:{} CE{:10.4} C.DIFF:{:10.4}"
                      .format(div, curr_energ, curr_energ - energ))
                # accept and continue (copy final function)
                _e = subprocess.call('cp fort.11 fort.10', shell=True)
                # increase the step for valid or deformation precision overflow 
                div = max(0, div - 2) ## smoothly recover the dq
                e_diff = curr_energ - energ
                energ = curr_energ
                b20_i = b20
                print("  * [OK] WF directly copied  [i{}]: DIV:{} DIFF{:10.4f} ENER{:10.4f} B{:5.3f}"
                      .format(i,div, e_diff, energ, b20_i))
            
            if prolate == 0:
                results.insert(0, res)
            else:
                results.append(res)   
            
            ## SECOND PROCESS --------------------------------
            if voidDD_path != None:
            
                # do a void step to activate DD with no rearrangement
                kwargs = {
                    Template.com : com_template.format(1,1),
                    Template.z   : z,
                    Template.a   : A,
                    Template.seed : 0,
                    Template.iterartions : 0,
                    Template.b20  : constr,
                    Template.hamil : 0,
                    Template.varN2: ""
                }
                res2 = _executeProgram(kwargs, output_filename+'_VS', b20,
                                       save_final_wf=False, noGradient=True)
                if res2 == None:
                    continue # dont save empty result
                if prolate == 0: #grow in order [-.5, -.4, ..., .0,..., +.4, +.5] 
                    results_voidStep.insert(0, res2)
                else:
                    results_voidStep.append(res2)          
                    
            print("-------------------------------------------------------------------------------")
            print()
            
            # intermediate print
            _exportResult(results, DataAxial.export_list_results)
            if voidDD_path != None:
                _exportResult(results_voidStep, voidDD_path) 

    # ## ------ end exec.  -----
    _exportResult(results, DataAxial.export_list_results)
    print("   ** generate File 1st convergence in:", DataAxial.export_list_results)
    if results_voidStep:
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


def mainLinux(z, n):
    #
    #%% Executing the process, run the list of isotopes
    #
    output_filename = 'aux_output' ### DataTaurus.output_filename_DEFAULT #
    A = n + z 
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataAxial.setUpFolderBackUp()
    if os.path.exists(DataAxial.export_list_results):
        os.remove(DataAxial.export_list_results)
    
    results = []
        
    print(HEAD)
    constr_N2, constr_DJ2, constr_MSR = '', '', ''
    
    
    deform_prolate = np.linspace(0.0, 40.0, num=45, endpoint=True)
    deform_oblate  = np.linspace(0.0,-40.0, num=45, endpoint=True) #18,
    for i_deform, deform in enumerate((deform_oblate, deform_prolate)):
        
        # create a spherical seed to proceed
        constr  = q20_constr_template.format(1,1, 0.000)
        # constr += b20_constr_template.format(0,1, b20_const/2)
        kwargs = {
            Template.com : com_template.format(1,1),
            Template.z   : z,
            Template.a   : A,
            Template.seed : 2,
            Template.iterartions : 2000,
            Template.b20  : constr,
            Template.hamil : 0,
            Template.varN2: constr_N2 + constr_DJ2 + constr_MSR
        }
        
        
        print(" * first convergence (seed2)")
        _ = _executeProgram(kwargs, output_filename, 0.0)
        print("   ... done.")
        
        ## ----- execution ----
        for q20_const in deform:
            # create a spherical seed to proceed
            #q20_const *= 2 * np.pi / (np.sqrt(5 * np.pi))
            constr  = q20_constr_template.format(1,1, q20_const)
            
            kwargs = {
                Template.com : com_template.format(1,1),
                Template.z   : z,
                Template.a   : A,
                Template.seed : 1,
                Template.iterartions : 2000,
                Template.b20   : constr,
                Template.hamil : 0,
                Template.varN2 : constr_N2 + constr_DJ2 + constr_MSR
            }
            
            res = _executeProgram(kwargs, output_filename, q20_const,
                                  print_result=False)
            if res == None:
                continue # dont save empty result
            
            # do a void step to activate DD with no rearrangement
            kwargs = {
                Template.com : com_template.format(1,1),
                Template.z   : z,
                Template.a   : A,
                Template.seed : 1,
                Template.iterartions : 500,
                Template.b20   : constr,
                Template.hamil : 0,
                Template.varN2 : constr_N2 + constr_DJ2 + constr_MSR
            }
            res = _executeProgram(kwargs, output_filename, q20_const)
            
            if i_deform == 0: #grow in order [-.5, -.4, ..., .0,..., +.4, +.5] 
                results.insert(0, res)
            else:
                results.append(res)          
                    
                    
            
    # ## ------ end exec.  -----
    data = []
    for res in results:
        # line = ', '.join([k+' : '+str(v) for k,v in res.__dict__.items()])
        line = res.getAttributesDictLike
        data.append(line+'\n')
    # for i, r in enumerate(results):
    #     print("{} : {},".format(i, r.r_isoscalar))
        
    with open(DataAxial.export_list_results, 'a+') as f:
        f.writelines(data)
        
    


#%% main
z = 12
n = 12
output_filename = 'aux_output'

tail = ''
# tail = 'B1'
# tail =  '1GCOM0'
# tail = 'B1COM0'
# tail = 'D1SnoR'
# tail = 'D1S_voidDD'
tail = 'D1S'

nucleus = []
for z in range(10,15, 2):
    for n in range(max(6, z-2), 17, 2):
        nucleus.append((z, n))


DataAxial.export_list_results = "export_PESz{}n{}Axial{}.txt".format(z,n,tail)
if __name__ == '__main__':
    output_filename = DataAxial.output_filename_DEFAULT
    
    if not os.getcwd().startswith('C:'):
        print()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(' Running PES with HFBAxial:', HAMIL_AXIAL_PROGRAM)
        print('  !!! CHECK, CHECK MZ:', HAMIL_AXIAL_PROGRAM.upper())
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print()
        
        for z, n in nucleus:
            print("PES for Z={} N={}".format(z,n))
            output_filename = 'aux_output'
            DataAxial.export_list_results = "export_PESz{}n{}Axial{}.txt".format(z,n, tail)
            voidDD_path = "export_PESz{}n{}Axial{}_voidDD.txt".format(z,n, tail)
            
            # mainLinux(z, n)
            # mainLinuxEvenlyDeform(z, n, -0.28, 0.28, 100, voidDD_path)
            # mainLinuxSecurePES(z, n, -0.30, 0.32, 100, 
            #                    voidDD_path=voidDD_path, b20_base= 0.3093)
            # mainLinuxSecurePES(z, n, -0.30, 0.30, 100, b20_base=-0.29)
            mainLinuxSweepingPES(z, n, -0.6, 0.6, 300, False, None)
    else:
        #%% process in windows
        
        results_axial = []
        import_file_Axi = 'BU_results_old/export_PESz{}n{}Axial{}.txt'.format(z, n, tail)
        with open(import_file_Axi, 'r') as f:
            data = f.readlines()
            for line in data:
                res = DataAxial(None, None, None, True)
                res.setDataFromCSVLine(line)
                results_axial.append(res)
        
        for attr_ in (
                      'E_HFB', 
                      'kin', 
                      'var_n', 'pair',#, 'properly_finished'
                      # 'Jx_var',
                      # 'Jz',
                      # 'r_isoscalar',
                      ):
            ## plot energies
            x_tau, y_tau = [], []
            x_ax,  y_ax  = [], []
            
            for r in results_axial:
                x_ax.append(r.q20_isoscalar)
                if attr_ == 'r_isoscalar':
                    y_ax.append(getattr(r, attr_, 0.0))#/(r.n + r.z))
                else:
                    y_ax.append(getattr(r, attr_, 0.0))#/(r.n + r.z))
            
            if attr_ == 'properly_finshed':
                y_ax = [1 if p == 'True' else 0 for p in y_ax]
            
            plt.figure()
            plt.xlabel(r"$Q_{20} [fm^2]$")
            plt.plot(x_ax,  y_ax,  'o-b',   label="HFB axial")
            plt.title(attr_+"  [Z:{} N:{}]  ".format(z, n)+" B1 no LS")
            plt.legend()
            plt.tight_layout()
            plt.show()
        