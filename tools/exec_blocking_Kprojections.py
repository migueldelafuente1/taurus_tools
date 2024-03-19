'''
Created on 19 mar 2024

@author: delafuente

Protocol to export the different projections of the K momenum in odd-even 
calculations:
    1. Perform a false ODD-EVEN calculaton
    2. Bock each state in the odd-particle sp-states and minimize again
        2.1 find for each K, in case of not finding it proceed with other seed
        2.2 In case K is not found, ignore K for that deformation
    3. Prepare and export all the components as a surface, also for a later PAV

'''
from .executors import ExeTaurus1D_DeformB20
from copy import deepcopy

class ExeTaurus1D_B20_OEblocking_Ksurfaces(ExeTaurus1D_DeformB20):
    '''
    
    '''
    IGNORE_BLOCKING = True

    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        ExeTaurus1D_DeformB20.__init__(self, z, n, interaction, *args, **kwargs)
        
        self._valid_Ks : list = []
        self._K_results : dict = {}
        self._K_seed_list : dict = {}
    
    def setInputCalculationArguments(self, core_calc=False, axial_calc=False, 
                                           spherical_calc=False, **input_kwargs):
        
        ExeTaurus1D_DeformB20.setInputCalculationArguments(self, 
                                                           core_calc=core_calc, 
                                                           axial_calc=axial_calc, 
                                                           spherical_calc=spherical_calc, 
                                                           **input_kwargs)
        _ = 0
    
    def setUpExecution(self, reset_seed=False, *args, **kwargs):
        ExeTaurus1D_DeformB20.setUpExecution(self, reset_seed=reset_seed, *args, **kwargs)
        
        self._valid_Ks = [k for k in range(-self._sp_2jmax, self._sp_2jmax+1, 2)]
        # skip invalid K for the basis, i.e. _sp_2jmin=3/2 -> ..., 5,3,-3,-5 ...
        self._valid_Ks = list(filter(lambda x: abs(x) >= self._sp_2jmin,
                                     self._valid_Ks))
        for k in self._valid_Ks:
            def_dct = list(map(lambda x: x[0], self._deformations_map[0]))
            def_dct+= list(map(lambda x: x[0], self._deformations_map[1]))
            def_dct.sort()
            def_dct = dict((kk, None) for kk in def_dct)
            
            self._K_results[k]   = def_dct
            self._K_seed_list[k] = deepcopy(def_dct)
            
        _ = 0
        
    def run(self):
        ExeTaurus1D_DeformB20.run(self)
        
        _ = 0
        # Perform the projections to save each K component
        for K in self._valid_Ks:
            pass
        
    
    def saveFinalWFprocedure(self, result, base_execution=False):
        ExeTaurus1D_DeformB20.saveFinalWFprocedure(self, result, base_execution=base_execution)
    