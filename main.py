'''
Created on Jan 10, 2023

@author: Miguel
'''
from scripts1d.beta_scripts import run_q20_surface
from tools.helpers import importAndCompile_taurus

from tools.hamiltonian_maker import TBME_HamiltonianManager



if __name__ == '__main__':
    
    exe_ = TBME_HamiltonianManager(1.69, 2, set_com2=True)
    exe_.setAndRun_D1Sxml()
    
    # importAndCompile_taurus()
    
    nucleus = [(2, 4), (4, 4)]
    run_q20_surface(nucleus, q_min=-5, q_max=5, N_max=10)