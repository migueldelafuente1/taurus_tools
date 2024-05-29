'''
Created on Jan 23, 2023

@author: Miguel
'''
import os
import shutil
from tools.Enums import OutputFileTypes, GognyEnum
from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.helpers import printf


def getInteractionFile4D1S(interactions, z,n, do_Coulomb=True, do_LS=True,
                           gogny_interaction = GognyEnum.D1S):
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
            MSG_ = "Arguments must be (MZmax <int>, (>=) Mzmin <int>, b_lengt <float>)"
            assert len(args) == 3, MSG_
            MZmax, MZmin, b_length = args
            
            if b_length == None:                    ## Semiempirical formula
                b_length = 1.005 * ((z+n)**(1/6))
            
            assert type(MZmax)==int and type(MZmin) == int and type(b_length)==float, MSG_
            assert MZmax >= MZmin and MZmin >= 0, "MZmax >= Mzmin >= 0"
            
            printf(f"  ** [] Generating Matrix Elements for D1S, zn={z},{n}, b={b_length:5.3f}"
                  f"  Major shells: [{MZmin}, {MZmax}]")
            exe_ = TBME_HamiltonianManager(b_length, MZmax, MZmin, set_com2=True)
            exe_.do_coulomb = do_Coulomb
            exe_.do_LS      = do_LS
            
            exe_.setAndRun_Gogny_xml(gogny_interaction)
            interaction = exe_.hamil_filename
            printf(f" ** [DONE] Interaction: [{interaction}]")
            
            return interaction
        
        else:
            raise Exception(f"Invalid interactions[z,n] types given: {interaction}")



def parseTimeVerboseCommandOutputFile(time_filename):
    """ 
    Process the /usr/bin/time -v <executable> output.
    Only gives times (real/cpu/system) and maximum ram used (extend for other values)
    """
    vals = {'user_time': 1,  'sys_time' : 2, 'real_time': 4,  'memory_max': 9}
    headers_checks = {
        'user_time': 'User time (seconds)',  
        'sys_time' : 'System time (seconds)', 
        'real_time': 'Elapsed (wall clock) time',  
        'memory_max': 'Maximum resident set size (kbytes)'}
    
    if not os.path.exists(time_filename):
        printf(f" [WARNING] Could not found timing file [{time_filename}]")
        return None
    
    aux = {}
    with open(time_filename, 'r') as f:
        lines = f.readlines()
        exit_status = int(lines[-1].replace('Exit status:', '').strip())
        if exit_status != 0:
            ## Error, the program prompt some error, returning None
            return aux
        for key_, indx in vals.items():
            if headers_checks[key_] not in lines[indx]:
                printf(f"[WARNING TIME OUTPUT PARSING] line [{indx}] for parameter [{key_}]",
                      f" does not match expected header [{headers_checks[key_]}].\n",
                      f"Got: [{lines[indx]}]")
            line = lines[indx].split(' ')[-1] # no argument after the last ":" has spaces
            
            if indx == 4: ## hh:mm:ss or mm:ss
                line = [float(x) for x in line.split(':')]
                if len(line) == 3: #has hours
                    line = 3600*line[0] + 60*line[1] + line[2] 
                else:
                    line = 60*line[0] + line[1]
            else:
                line = int(line) if indx == 9 else float(line) 
         
            aux[key_] = line
    return aux

