'''
Created on Jan 23, 2023

@author: Miguel
'''
import os
import shutil
from tools.Enums import OutputFileTypes
from tools.hamiltonianMaker import TBME_HamiltonianManager



def getInteractionFile4D1S(interactions, z,n):
    """
    This function import a certain hamil file for the calculations to run:
    
    Procedure:
        * if interactions is <str>:
            It returns or copy the hamiltionian files (in case is in a folder)
        * if interaction is a <dict of int tuples>: {(z,n): 'hamil_1', ...}
            It returns the hamiltonian files (in case also of a folder)
        
        (NOTE): interactions in folders can not be nested, mandatory: fld/hamil.sho etc
        
        * if generate_hamils=True and interactions are a dictionary following
          the format:
            {(z,n): (MZmax <int>, Mzmin <int>, b_lengt <float>)}
            It calls the TBME_HamiltonianManager and runs full D1S (LS+BB+Coul)
    """
    
    if isinstance(interactions, (str, dict)):
        if isinstance(interactions, dict):
            interaction = interactions[(z, n)]
        else:
            interaction = interactions ## common hamiltonian
        
        if type(interaction) == str:
            
            if '/' in interaction:
                args = interaction.split('/')
                if len(args) != 2:
                    raise Exception("do not nest the hamiltonian files for the folder", args)
                ext_fn = [(ext, os.path.exists(interaction+ext)) 
                          for ext in OutputFileTypes.members()]
                files_ = dict(filter(lambda x: x[1], ext_fn))
                
                interaction = args[1]
                for ext, filesrc in files_.items():
                    shutil.copy(filesrc, interaction+ext)
                
            return interaction
        
        elif isinstance(interaction, tuple):
            ## TODO: assert format (MZmax, Mzmin, b_length)
            args = interactions[(z, n)]
            MSG_ = "args must be (MZmax <int>, (>=) Mzmin <int>, b_lengt <float>)"
            assert len(args) == 3, MSG_
            MZmax, MZmin, b_length = args
            assert type(MZmax)==int and type(MZmin) == int and type(b_length)==float, MSG_
            assert MZmax >= MZmin and MZmin >= 0, "MZmax >= Mzmin >= 0"
            
            print(f"  Generating Matrix Elements for D1S, zn={z},{n}, b={b_length:5.3f}")
            exe_ = TBME_HamiltonianManager(b_length, MZmax, MZmin, set_com2=True)
            exe_.setAndRun_D1Sxml()
            interaction = exe_.hamil_filename
            print(f" [DONE] Interaction: {interaction}")
            
            return interaction
        
        else:
            raise Exception(f"Invalid interactions[z,n] types given: {interaction}")
            
            
            
            
    