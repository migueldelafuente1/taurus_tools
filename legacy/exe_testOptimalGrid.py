'''
Created on Apr 19, 2022

@author: Miguel
'''
from builtins import list
from collections import OrderedDict
from datetime import datetime
import os
import shutil
import subprocess

from exe_isotopeChain_taurus import DataTaurus
import matplotlib.pyplot as plt
import numpy as np


class Template:
    com = 'com'
    z   = 'z'
    n   = 'n'
    seed= 'seed'
    b20 = 'b20'
    gamma = 'gamma'
    steps = 'steps'
    


TEMPLATE_INP = """Interaction   
-----------
Master name hamil. files      D1S_t0_SPSD
Center-of-mass correction     {com}
Read reduced hamiltonian      0
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
Maximum no. of iterations     {steps} 
Step intermediate wf writing  0
More intermediate printing    0
Type of gradient              1
Parameter eta for gradient    0.050E-00
Parameter mu  for gradient    0.300E-00
Tolerance for gradient        1.00E-03

Constraints             
-----------
Force constraint N/Z          1
Constraint beta_lm            2
Pair coupling scheme          1
Tolerance for constraints     1.000E-06
Constraint multipole Q10      1   0.000
Constraint multipole Q11      1   0.000
Constraint multipole Q20      {b20}
Constraint multipole Q21      1   0.000
Constraint multipole Q22      {gamma}
Constraint multipole Q30      0   0.000
Constraint multipole Q31      1   0.000
Constraint multipole Q32      1   0.000
Constraint multipole Q33      1   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      1   0.000
Constraint multipole Q42      1   0.000
Constraint multipole Q43      1   0.000
Constraint multipole Q44      1   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     1   0.000
Constraint pair P_T00_J1m1    1   0.000
Constraint pair P_T00_J1p1    1   0.000
Constraint pair P_T10_J00     1   0.000
Constraint pair P_T1m1_J00    0   0.000
Constraint pair P_T1p1_J00    0   0.000
Constraint field Delta        0   0.000
"""

TEMPLATE_DD_INP = """* Density dep. Interaction:    ------------
eval_density_dependent (1,0)= 1
eval_rearrangement (1,0)    = 1
t3_DD_CONST [real  MeV]     = 1.390600d+03    !1.000000d+03    !
x0_DD_FACTOR                = 1.000000d+00
alpha_DD                    = 0.333333d+00    !1.000000d+00    !
* Integration parameters:      ------------
*  0 trapezoidal, 1 Gauss-Legendre, 2 Gauss-Laguerre(r)/Legendre, 3 Laguerre-Lebedev
integration_method (0,1,2,3)= 3
export_density (1, 0)       = 1
r_dim                       = {r_dim}
Omega_Order                 = {omega_order}
THE_grid                    = 10
PHI_grid                    = 10
R_MAX                       = 08.500000d+00
* Integration parameters:      ------------
"""



HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
uncoupled_DD_FILE = 'uncoupled_DD.2b'

def preconvergeSolution(z, n, r_dim = 12, omega_order=10, beta=0.0, gamma=0.0):
    output_filename = DataTaurus.output_filename_DEFAULT
    try:
        status_fin = ''
        
        ## ----- execution ----
        
        kwargs = {
            Template.com : 1,
            Template.z   : z,
            Template.n   : n,
            Template.seed : 3,
            Template.steps : 1000,
            Template.b20    : "1 {:5.4}".format(beta),
            Template.gamma  : "1 {:5.4}".format(gamma)
        }
        
        #%% Set the first convergence step
        text = TEMPLATE_INP.format(**kwargs)
        with open(DataTaurus.INPUT_FILENAME, 'w+') as f:
            f.write(text)
            
        kwargs = {
            'r_dim'       : r_dim,
            'omega_order' : omega_order
        }
        # TODO: Uncomment
        text = TEMPLATE_DD_INP.format(**kwargs)
        with open('input_DD_PARAMS.txt', 'w+') as f:
            f.write(text)
        
        _e = subprocess.call('./taurus_vap.exe < {} > {}'
                                  .format(DataTaurus.INPUT_FILENAME, 
                                          output_filename), 
                              shell=True,
                             timeout=8640) # 1 day timeout)
        res = DataTaurus(z, n, output_filename)        
        
        ## Save files in buck up
        _e = subprocess.call('mv {} {}'.format(output_filename, 
                                  os.getcwd()+'/'+FOLDER_DUMP
                                  +'/'+output_filename[:-3]
                                  + '_Z{}N{}'.format(z,n)
                                  + '_Rd{}Od{}_1st.2b'.format(r_dim, omega_order)),
            #+'_{}'.format(str(int(1000*b20_const)).replace('-','neg'))),
                             shell=True)
        _e = subprocess.call('mv *.dat *.red {}/'.format(FOLDER_DUMP) , shell=True)
        _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
        _e = subprocess.call('cp final_wf.bin {} {}/'
                                .format(output_filename, FOLDER_DUMP), shell=True)
        _e = subprocess.call('cp uncoupled_DD.2b {}/uncoupled_DD(1st).2b'
                                .format(FOLDER_DUMP), shell=True)
        _e = subprocess.call('cp DIMENS_indexes_and_rhoLRkappas.txt {}/'
                                .format(FOLDER_DUMP), shell=True)
                
        if not res.properly_finished:
            status_fin += 'X'
        else:
            status_fin += '.'
              
        print(" {:2} {:2}  ( {})    {:9.4f}  {:9.4f}  {:7.4f}  {:5.4f}"
              .format(z, n, status_fin, res.E_HFB, res.kin, res.pair, beta))
        print(" >>", datetime.now().time())
    except Exception as e:
        print(" >> EXCEP >>>>>>>>>>  ")
        print(" >> ", beta)
        print(" > ", e, "<")
        print(res)
        print(" << EXCEP <<<<<<<<<<  ")
        
    

def evalGrid(z, n, r_dim, omega_order, beta, gamma):
    #%% Executing the process, run the list of isotopes
    #
    output_filename = DataTaurus.output_filename_DEFAULT
    results = []
    
    # print(HEAD)
    
    try:
        status_fin = ''
                
        kwargs = {
            Template.com : 1,
            Template.z   : z,
            Template.n   : n,
            Template.seed  : 1,
            Template.steps : 0, 
            Template.b20   : "1 {:5.4}".format(beta),
            Template.gamma : "1 {:5.4}".format(gamma)
        }
        
        #%% Set the first convergence step
        text = TEMPLATE_INP.format(**kwargs)
        with open(DataTaurus.INPUT_FILENAME, 'w+') as f:
            f.write(text)
            
        kwargs = {
            'r_dim'       : r_dim,
            'omega_order' : omega_order
        }
        # TODO: Uncomment
        text = TEMPLATE_DD_INP.format(**kwargs)
        with open('input_DD_PARAMS.txt', 'w+') as f:
            f.write(text)
        
        _e = subprocess.call('./taurus_vap.exe < {} > {}'
                                  .format(DataTaurus.INPUT_FILENAME, 
                                          output_filename), 
                              shell=True,
                             timeout=86400) # 1 day timeout)
        res = DataTaurus(z, n, output_filename)
        results.append(res)
        #res.getDDEnergyEvolution()
        
        ## TODO UNCOMMENT
        _e = subprocess.call('mv {} {}'.format(uncoupled_DD_FILE, 
                                  os.getcwd()+'/'+FOLDER_DUMP
                                  +'/'+uncoupled_DD_FILE[:-3]
                                  + '_Z{}N{}'.format(z,n)
                                  + '_Rd{}Od{}.2b'.format(r_dim, omega_order)),
            #+'_{}'.format(str(int(1000*b20_const)).replace('-','neg'))),
                             shell=True)
        _e = subprocess.call('rm *.dat *.red', shell=True)
                
        if not res.properly_finished:
            status_fin += 'X'
        else:
            status_fin += '.'
              
        # print(" {:2} {:2}  ( {})    {:9.4f}  {:9.4f}  {:7.4f}  {:5.4f}"
        #       .format(z, n, status_fin, res.E_HFB, res.kin, res.pair, beta))
        # print(" >>", datetime.now().time())
    except Exception as e:
        print(" >> EXCEP >>>>>>>>>>  ")
        print(" > ", e, "<")
        print(" << EXCEP <<<<<<<<<<  ")
            

FOLDER_DUMP = 'GridConvergence'

if __name__ == '__main__':
    
    z = 14
    n = 12
    output_filename = 'aux_output'
    DataTaurus.export_list_results = "export_z{}n{}TaurusGrid.txt".format(z,n)
    
    beta  = 0.15
    gamma = 40.0
    
    FOLDER_DUMP = 'GridConvergenceZ{}N{}_B{}G{}'.format(z,n,beta,gamma)
    FOLDER_DUMP = FOLDER_DUMP.replace('.','')
    DataTaurus.BU_folder = FOLDER_DUMP
    
    
    Rsteps, Omsteps = 15, 16
    
    
    if not os.getcwd().startswith('C:'):    #
        # Overwrite/create the buck up folder
        DataTaurus.setUpFolderBackUp()
        
        print("1st convergence to beta =", beta,' gamma=', gamma)
        print("start:", datetime.now().time())
        preconvergeSolution(z, n, omega_order=15, beta=beta, gamma=gamma)
        print("end:", datetime.now().time())
        i = 1
        for r_dim in [3+i for i in range(Rsteps)]:
            for omega_order in [3+j for j in range(Omsteps)]:
                print(" ({} /{})  Rdim={} ANGdim={}"
                      .format(i,Rsteps*Omsteps, r_dim, omega_order))
            
                evalGrid(z, n, r_dim, omega_order, beta, gamma)
                i += 1
        print("end:", datetime.now().time())
                
    else:
        #%% print the results.
        data = {}
        r_dim = set()
        a_dim = set()
        states_set = set()
        
        def __get_line_file(sp_line):
            a, b, c, d, vDD = sp_line.split()
            a,b,c,d = int(a), int(b), int(c), int(d)
            vDD = float(vDD)
            return a, b, c, d, vDD
        
        
        for file_ in sorted(list(filter(lambda f: '.2b' in f,#f.endswith('.2b'), 
                                 os.listdir(FOLDER_DUMP)))):
            
            r, ang = file_[:-3].split('_')[3].replace('Rd','').split('Od')
            r, ang = int(r), int(ang)
            r_dim.add(r)
            a_dim.add(ang)
            if r in data:
                data[r][ang] = {}
            else:
                data[r] = {ang : {}}
            
            # print(file_, r, ang)
            with open(FOLDER_DUMP+'/'+file_, 'r') as f:
                aux_data =  f.readlines()
                HOsp_dim = int(aux_data[0].split('=')[1])
                
                for sp_line in aux_data[1:]:
                    a, b, c, d, vDD = __get_line_file(sp_line)
                    data[r][ang][a,b,c,d] = vDD
                    states_set.add((a,b,c,d))
                    
        r_dim = list(r_dim)
        a_dim = list(a_dim)    
        #TODO : Print a single matrix element
        
        state_ = (1, 22, 1,22)
        state_ = (3, 22, 7,21)
        # state_ = (12,31,12,38)
        # state_ = (12,21,16,31)
        plt.figure()
        for ang in a_dim[1:-1:1]:
            x, vDD = [], []
            for r in r_dim[1:]:
                if not ang in data[r]:
                    continue
                vDD.append(data[r][ang].get(state_, None))
                x.append(r)
            plt.plot(x, vDD,'.-', label=str(ang) )
        
        plt.legend()
        plt.xlabel('R DIM')
        plt.title("<{} {}| v |{} {}> Evolution".format(*state_))
        plt.show()
        
        # TODO: Evaluate a median or an average of the matrix elements on the grid.
        # Fix a range of R dim (reasonable) and get the difference between MIN MAX
        # for all Angular grids, then plot the differences over a tolerance
        TOL_INTEGR = 1.0e-7
        R_min  = 4
        Om_min = 4
        
        states_diff = dict([(st_, None) for st_ in states_set])
        
        diff_plot = []
        st_plot   = []
        overTol = 0
        for i, st_ in enumerate(states_set):
            max_ = -1000
            min_ = 1000
            for ang in a_dim[Om_min:]:
                for r in r_dim[R_min:]:
                    if not ang in data[r]:
                        continue
                    max_ = max(data[r][ang][st_], max_)
                    min_ = min(data[r][ang][st_], min_)
            diff = abs(max_ - min_)
            
            states_diff[st_] = diff
            
            if diff > TOL_INTEGR:
                st_plot.append(i)
                diff_plot.append(diff)
                overTol += 1
        
        plt.figure()
        plt.plot(st_plot, diff_plot, 'b'
                 )
        plt.title("Maximum difference on (Radial >{}/Angular >{}) margins "
                  .format(R_min-1, Om_min-1)+
                  " OVER {}:[{} /{}]".format(TOL_INTEGR, overTol, len(states_set)))
        plt.ylabel('max difference [MeV]')
        plt.xlabel('states (enumerated)')
        plt.show()
                
        
        
        
                
                    
                    
         
        
        
        
        
        
        
        