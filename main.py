'''
Created on Jan 10, 2023

@author: Miguel
'''
import os
from scripts1d.beta_scripts import run_q20_surface, run_b20_surface
from tools.helpers import importAndCompile_taurus, TBME_SUITE

from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus

if not (InputTaurus.PROGRAM in os.listdir()):
    importAndCompile_taurus()

if __name__ == '__main__':
    
    exe_ = TBME_HamiltonianManager(1.75, 4, set_com2=True)
    exe_.setAndRun_D1Sxml()
    
    # importAndCompile_taurus()
    
    nucleus = [(2, 4), (4, 4)]
    interactions = {(2,4): (4,0, 2.0), (4,4): 'hamil_2'}
    run_b20_surface(nucleus, q_min=-1., q_max=1., N_max=10)
    