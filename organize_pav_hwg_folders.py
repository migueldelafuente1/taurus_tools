'''
Created on 16 may 2024

@author: delafuente

script to preparate the folders for PAV and/or HWG folders from different BU_folders

The naming and scripts are the automatized sorting from taurus_tools 

Using the scripts and objects for input for the different programs
'''

import os
import shutil

from tools.exec_blocking_Kprojections import _SlurmJob1DPreparation
from tools.inputs import InputTaurus, InputTaurusPAV, InputTaurusMIX
from tools.helpers import importAndCompile_taurus

def sort_by_deformation_naming(def_list):
    """ sorting as [_0.8.bin, _0.6.bin, .. , 0.2.bin, 0.4.bin ...]"""
    def_list = list(def_list)
    pos_vals = filter(lambda x: not x.startswith('_'), def_list)
    neg_vals = filter(lambda x:     x.startswith('_'), def_list)
    
    sorted_list = sorted(list(neg_vals), reverse=True) + sorted(list(pos_vals))
    
    return sorted_list

#===============================================================================
# SCRIPTS
#===============================================================================

def basic_eveneven_mix_from_vap(MAIN_FLD):
    """
    
    """
    
    assert os.path.exists(MAIN_FLD), "Main folder does not exists."
    os.chdir(MAIN_FLD)
    # employs the seed_* sorted
    pass

def oddeven_mix_same_K_from_vap(K, MAIN_FLD, interaction, nuclei, 
                                PNP_fomenko=1, Jmax=1):
    """
    Create all the folders with executable, hamil, etc.
        1, 
        2, 
        ,...
        N:[left_wf.bin, right_wf.bin, hamil.* taurus_pav.exe, input_pav.exe]
        
        scripts: 
    
    """
    
    if not os.path.exists(MAIN_FLD): 
        print("Main folder does not exists.", MAIN_FLD)
        return
    RETURN_FLD = "/".join([".." for _ in MAIN_FLD.split('/')])
    
    if MAIN_FLD:
        shutil.copy('taurus_pav.exe', MAIN_FLD)
        shutil.copy('taurus_mix.exe', MAIN_FLD)
    
    os.chdir(MAIN_FLD)
    TEMP_BU      = "BU_folder_{}_z{}n{}/"
    BU_KVAP      =f"{K}_0_VAP/"
    DEST_FLD     = 'PNPAMP_HWG/'
    DEST_FLD_HWG = 'HWG'
    valid_J_list = [i for i in range(1, Jmax+1, 2)]
    
    input_pav_args = {
        InputTaurusPAV.ArgsEnum.n_Mphi: PNP_fomenko,
        InputTaurusPAV.ArgsEnum.z_Mphi: PNP_fomenko,
        InputTaurusPAV.ArgsEnum.disable_simplifications_JMK: True,
        InputTaurusPAV.ArgsEnum.alpha: 15,
        InputTaurusPAV.ArgsEnum.beta:  20,
        InputTaurusPAV.ArgsEnum.gamma: 15,
        InputTaurusPAV.ArgsEnum.disable_simplifications_NZA: True,
        InputTaurusPAV.ArgsEnum.j_min: K,
        InputTaurusPAV.ArgsEnum.j_max: Jmax,
        InputTaurusPAV.ArgsEnum.com: 1,
        # InputTaurusPAV.ArgsEnum.empty_states:   1,
        # InputTaurusPAV.ArgsEnum.cutoff_overlap: 1.0e-9,
    }
    input_mix_args = {
        InputTaurusMIX.ArgsEnum.j_val : 1, 
        InputTaurusMIX.ArgsEnum.opt_convergence_analysis: 1,
        InputTaurusMIX.ArgsEnum.opt_remove_neg_eigen: 1,
        InputTaurusMIX.ArgsEnum.max_energy: 30,
        InputTaurusMIX.ArgsEnum.j_val : 1,
        InputTaurusMIX.ArgsEnum.parity: 1,
        #
        InputTaurusMIX.CutoffArgsEnum.cutoff_overlap: 1.0e-9,
        InputTaurusMIX.CutoffArgsEnum.cutoff_energy : 0.0,
        InputTaurusMIX.CutoffArgsEnum.cutoff_JzJ2   : 1.0e-6,
        InputTaurusMIX.CutoffArgsEnum.cutoff_norm_eigen: 1.0e-6,
        InputTaurusMIX.CutoffArgsEnum.cutoff_negative_eigen: 1.0e-6
    }
    
    for z, n in nuclei:
        print(" Starting for", TEMP_BU.format(interaction, z, n))
        os.chdir(TEMP_BU.format(interaction, z, n))
        #os.mkdir(DEST_FLD)
        bins_ = os.listdir(BU_KVAP)
        bins_ = filter(lambda x: x.endswith('.bin'), bins_)
        bins_ = sort_by_deformation_naming(bins_)
        print("  1. creating folders and common files: ", BU_KVAP, bins_)
        
        pav_obj = InputTaurusPAV(z, n, interaction, **input_pav_args)
        with open('input_pav.txt', 'w+') as f:
            f.write(pav_obj.getText4file())
        
        # clear FLD if exists
        print("  2. Creating SLUM folder:", DEST_FLD) 
        if os.path.exists(DEST_FLD):
            shutil.rmtree(DEST_FLD)
        os.mkdir(DEST_FLD)
        os.mkdir(DEST_FLD + DEST_FLD_HWG)
        
        # copy pav folders.
        print("  3. Copying binaries, etc for PAV") 
        bins2copy = []
        for i, b1 in enumerate(bins_):
            for i2 in range(i, len(bins_)):
                bins2copy.append( (b1, bins_[i2]) )
        for i, l_r_wf in enumerate(bins2copy):
            fld_i = DEST_FLD+str(i+1)+'/'
            os.mkdir(fld_i)
            shutil.copy(BU_KVAP+l_r_wf[0], fld_i+'left_wf.bin')
            shutil.copy(BU_KVAP+l_r_wf[1], fld_i+'right_wf.bin')
            for tail_ in ('.2b', '.com', '.sho'):
                shutil.copy(interaction+tail_, fld_i)
            shutil.copy('input_pav.txt', fld_i)
            shutil.copy('../taurus_pav.exe', fld_i)
            os.chmod(fld_i+'/taurus_pav.exe', 0o777)
        
        # hwg stuff
        print("  3. Preparing MIX folder.") 
        shutil.copy('../taurus_mix.exe', DEST_FLD+'/'+DEST_FLD_HWG)
        mix_obj = InputTaurusMIX(z, n, len(bins2copy), **input_mix_args)
        with open(DEST_FLD+'/'+DEST_FLD_HWG+'/input_mix.txt', 'w+') as f:
            f.write(mix_obj.getText4file())
        
        # create the scripts
        print("  4. Creating SLURM job scripts.") 
        slurm = _SlurmJob1DPreparation(interaction, len(bins_), valid_J_list,
                                       PAV_input_filename='input_pav.txt',
                                       HWG_input_filename='input_mix.txt')       
        
        for fn_, scr_ in slurm.getScriptsByName().items():
            with open(DEST_FLD+'/'+fn_, 'w+') as f:
                f.write(scr_)
        
        ## done, go back
        print(" Done.")
        os.chdir('..')
    
    os.chdir(RETURN_FLD)
    print("## Script has ended for K=", K)
        


if __name__ == '__main__':
    
    ## Exists the programs and compile with ifort.
    if False:
        importAndCompile_taurus(use_dens_taurus=False, 
                                pav = not os.path.exists('taurus_pav.exe'), 
                                mix = not os.path.exists('taurus_mix.exe'))
    
    ## TESTING_
    # inter  = 'B1_MZ3' 
    # nuclei = [(2, 1), (2, 3)]
    # oddeven_mix_same_K_from_vap(1, 'results', inter, nuclei, 
    #                             PNP_fomenko=7, Jmax=9, )
    #
    # raise Exception("STOP-TEST")
    ## 
    inter  = 'B1_MZ4' 
    nuclei = [(12, 11+ 2*i) for i in range(0, 5)]
    # nuclei = [(15, 8+ 2*i)  for i in range(0, 5)]
    # nuclei = [(17, 12+ 2*i) for i in range(0, 5)]
    
    for K in (1, 3, 5):
        MAIN_FLD = f'results/MgK{K}'
        oddeven_mix_same_K_from_vap(K, MAIN_FLD, inter, nuclei,
                                    PNP_fomenko=7, Jmax=9, )

