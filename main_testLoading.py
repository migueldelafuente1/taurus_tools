'''
Created on Mar 3, 2023

@author: Miguel
'''
from scripts1d.loadingAndTiming_scripts import run_IterTimeAndMemory_from_Taurus_byShellsAndIntegrationMesh

if __name__ == '__main__':
    Z,N = 12,10
    
    # RO = (6, 6)
    # RO = (10,10)
    RO = (13,13)
    # RO = (16,16)
    # RO = (20,20)
    
    run_IterTimeAndMemory_from_Taurus_byShellsAndIntegrationMesh(
        Mzmax=7, ROmegaMax=RO, z_numb=Z, n_numb=N)