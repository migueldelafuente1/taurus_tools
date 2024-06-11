'''
Created on 16 may 2024

@author: delafuente

script to preparate the folders for PAV and/or HWG folders from different BU_folders

The naming and scripts are the automatized sorting from taurus_tools 

Using the scripts and objects for input for the different programs
'''

import os
import shutil
from pathlib import Path

from tools.exec_blocking_Kprojections import _SlurmJob1DPreparation
from tools.inputs import InputTaurus, InputTaurusPAV, InputTaurusMIX
from tools.helpers import importAndCompile_taurus, elementNameByZ

def sort_by_deformation_naming(def_list):
    """ sorting as [_0.8.bin, _0.6.bin, .. , 0.2.bin, 0.4.bin ...]"""
    def_list = list(def_list)
    pos_vals = filter(lambda x: not x.startswith('_'), def_list)
    neg_vals = filter(lambda x:     x.startswith('_'), def_list)
    
    sorted_list = sorted(list(neg_vals), reverse=True) + sorted(list(pos_vals))
    
    return sorted_list

input_pav_args = {
    # InputTaurusPAV.ArgsEnum.n_Mphi: PNP_fomenko,
    # InputTaurusPAV.ArgsEnum.z_Mphi: PNP_fomenko,
    InputTaurusPAV.ArgsEnum.disable_simplifications_JMK: True,
    InputTaurusPAV.ArgsEnum.alpha: 15,
    InputTaurusPAV.ArgsEnum.beta:  20,
    InputTaurusPAV.ArgsEnum.gamma: 15,
    InputTaurusPAV.ArgsEnum.disable_simplifications_NZA: True,
    # InputTaurusPAV.ArgsEnum.j_min: min(K_list),
    # InputTaurusPAV.ArgsEnum.j_max: max(max(K_list), Jmax),
    InputTaurusPAV.ArgsEnum.com: 1,
    InputTaurusPAV.ArgsEnum.empty_states:   0,
    # InputTaurusPAV.ArgsEnum.cutoff_overlap: 1.0e-9,
}
input_mix_args = {
    InputTaurusMIX.ArgsEnum.opt_convergence_analysis: 1,
    InputTaurusMIX.ArgsEnum.opt_remove_neg_eigen: 1,
    InputTaurusMIX.ArgsEnum.max_energy: 30,
    InputTaurusMIX.ArgsEnum.j_val : 'J_VAL',
    InputTaurusMIX.ArgsEnum.parity: 1,
    #
    InputTaurusMIX.CutoffArgsEnum.cutoff_overlap: 1.0e-9,
    InputTaurusMIX.CutoffArgsEnum.cutoff_energy : 0.0,
    InputTaurusMIX.CutoffArgsEnum.cutoff_JzJ2   : 1.0e-6,
    InputTaurusMIX.CutoffArgsEnum.cutoff_norm_eigen: 1.0e-6,
    InputTaurusMIX.CutoffArgsEnum.cutoff_negative_eigen: 1.0e-6
}
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
                                PNP_fomenko=1, Jmax=1, RUN_SBATCH=False):
    """
    Create all the folders with executable, hamil, etc.
        1, 
        2, 
        ,...
        N:[left_wf.bin, right_wf.bin, hamil.* taurus_pav.exe, input_pav.exe]
        
        scripts: 
    
    """
    print("## Script begins for K=", K, ' *************************** \n')
    if not os.path.exists(MAIN_FLD): 
        print(" [ERROR] Main folder does not exists:", MAIN_FLD)
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
    
    global input_pav_args 
    input_pav_args[InputTaurusPAV.ArgsEnum.n_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.z_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.j_min]  = K
    input_pav_args[InputTaurusPAV.ArgsEnum.j_max]  = Jmax
    
    global input_mix_args
    
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
        for tail_ in ('.2b', '.com', '.sho'):
            shutil.copy(interaction+tail_, DEST_FLD)
        os.mkdir(DEST_FLD + DEST_FLD_HWG)
        
        # copy pav folders.
        print("  3. Copying binaries, etc for PAV") 
        bins2copy = []
        k, gcm_files = 0, {'gcm': [], 'gcm_diag': [], 'gcm_3': []}
        for i, b1 in enumerate(bins_):
            for i2 in range(i, len(bins_)):
                k += 1
                bins2copy.append( (b1, bins_[i2]) )
                gcm_files['gcm_3'].append(f"    {i+1: <6}")
                gcm_files[  'gcm'].append(f"{b1: <15}    {bins_[i2]: <15}    {i+1: <4}  {i2+1: <4}")
                if i2 == i: gcm_files['gcm_diag'].append(f"    {k: <6}")
        
        for i, l_r_wf in enumerate(bins2copy):
            fld_i = DEST_FLD+str(i+1)+'/'
            os.mkdir(fld_i)
            shutil.copy(BU_KVAP+l_r_wf[0], fld_i+'left_wf.bin')
            shutil.copy(BU_KVAP+l_r_wf[1], fld_i+'right_wf.bin')
            
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
            dst_fld = DEST_FLD if fn_ != 'hw.x' else DEST_FLD+'/'+DEST_FLD_HWG
            with open(dst_fld+'/'+fn_, 'w+') as f: f.write(scr_)
        for fn_, scr_ in gcm_files.items():
            with open(DEST_FLD+'/'+fn_, 'w+') as f: f.write("\n".join(scr_))
        
        # run sbatch
        if RUN_SBATCH: 
            os.chdir(DEST_FLD)
            os.system('sbatch sub_1.x')
            os.chdir('..')
        
        ## done, go back
        print(" Done.")
        os.chdir('..')
    
    os.chdir(RETURN_FLD)
    print("## Script has ended for K=", K, '\n')
    

def oddeven_mix_multiK_from_sameFld_vap(K_list, MAIN_FLD_TMP, interaction, nuclei,
                                        PNP_fomenko=1, Jmax=1, RUN_SBATCH=False):
    """
    Create all the folders with executable, hamil, etc. In this case, using all
    the K states blocked for the nuclei.
        1, 
        2, 
        ,...
        N:[left_wf.bin, right_wf.bin, hamil.* taurus_pav.exe, input_pav.exe]
        
        scripts: 
    
    Args:
    MAIN_FLD_TMP is a template for the placing of the different BU_folder_{inter}_zn:
        
    MULTIK_SAME_FLD: if the folder contains, for each nuclei all the Ks, set True
        otherwise, K is being read from different folders.
        * TRUE
    
    """
    
    K_list = sorted(K_list)
    print("## Script begin for the list K:", K_list, " ***************** \n")    
    
    global input_pav_args 
    input_pav_args[InputTaurusPAV.ArgsEnum.n_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.z_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.j_min]  = min(K_list)
    input_pav_args[InputTaurusPAV.ArgsEnum.j_max]  = max(max(K_list), Jmax)
    
    global input_mix_args
    
    TEMP_BU      = "BU_folder_{}_z{}n{}/"
    DEST_FLD     = 'kmix_PNPAMP_z{}n{}/'
    DEST_FLD_HWG = 'HWG'
    valid_J_list = [i for i in range(1, Jmax+1, 2)]
    
    first_step = True
    nuclei_by_K_found = {}
    path_migration, fld_migration = {}, {}
    # copy pav folders.
    print("  1. Creating folders, Copying binaries, etc for PAV/HWG") 
    for K in K_list:
        print("  K =", K, "  *first_state =", first_step)
        BU_KVAP      =f"{K}_0_VAP/"
        
        MAIN_FLD = Path(MAIN_FLD_TMP.format(K=K))
        if not MAIN_FLD.exists(): 
            print(" [ERROR] Main folder does not exists:", MAIN_FLD)
            return
        
        MAIN_DEST_PATH = MAIN_FLD.parent
        
        if first_step:
            ## if previous k-mix folders clear
            for fld_ in filter(lambda x: os.path.isdir(x)
                                         and x.startswith('kmix_PNPAMP_z'), 
                               os.listdir(MAIN_DEST_PATH)):
                fld_pav = Path(MAIN_DEST_PATH) / fld_
                if fld_pav.exists(): shutil.rmtree(fld_pav)
            
        for z, n in nuclei:
            fld1 = MAIN_FLD / Path(TEMP_BU.format(interaction, z, n))
            
            fld_pav = MAIN_DEST_PATH / DEST_FLD.format(z, n)
            fld_mix = fld_pav / DEST_FLD_HWG
            
            if first_step:                 
                fld_pav.mkdir(parents=True, exist_ok=True)
                fld_mix.mkdir(parents=True, exist_ok=True)
                shutil.copy('taurus_pav.exe', fld_pav)
                shutil.copy('taurus_mix.exe', fld_mix)
                
                pav_obj = InputTaurusPAV(z, n, interaction, **input_pav_args)                
                mix_obj = InputTaurusMIX(z, n, 0, **input_mix_args)
                
                with open(fld_pav / 'input_pav.txt', 'w+') as f:
                    f.write(pav_obj.getText4file())
                with open(fld_mix / 'input_mix.txt', 'w+') as f:
                    f.write(mix_obj.getText4file())
                
                for tail_ in ('.2b', '.com', '.sho'):
                    shutil.copy(fld1 / (interaction+tail_), fld_pav)
            
            
            ## The files migration for each k.
            if not (z,n) in fld_migration:  fld_migration[(z,n)]  = (fld_pav, fld_mix)
            if not (z,n) in path_migration: path_migration[(z,n)] = []
            if not (z,n) in nuclei_by_K_found: nuclei_by_K_found[(z,n)] = {}
            fld_kvap = fld1 / BU_KVAP 
            if not fld_kvap.exists(): print(f" [Error] K={K} folder not present {z},{n}")
            nuclei_by_K_found[(z,n)][K] = fld_kvap.exists()
            
            def_list = filter(lambda x: x.endswith('.bin'), os.listdir(fld_kvap))
            def_list = sort_by_deformation_naming(def_list)
            for bin_ in def_list:
                bin2_ = bin_.replace('.bin', f'_{K}.bin')
                path_migration[(z,n)].append(bin2_)
                shutil.copy(fld_kvap / bin_, fld_pav / bin2_)
        
        if first_step: first_step = False
    
    _ = 0
    # copy pav folders.
    print("  2. Copying binaries, etc for PAV") 
    for zn, k_founds in nuclei_by_K_found.items():
        
        fld_pav, fld_mix = fld_migration[zn]
        bins_ = path_migration[zn]
        
        if not all(k_founds.values()):
            print(f" [ERROR] Non all K were present for z,n={zn}, skipping ,{k_founds}")
        
        os.chdir(fld_pav)
        
        _pth_separ = '\\' if os.getcwd().startswith('C:') else '/'
        RETURN_FLD = "/".join([".." for _ in str(fld_pav).split(_pth_separ)])
        
        bins2copy = []
        k, gcm_files = 0, {'gcm': [], 'gcm_diag': [], 'gcm_3': []}
        for i, b1 in enumerate(bins_):
            for i2 in range(i, len(bins_)):
                k += 1
                bins2copy.append( (b1, bins_[i2]) )
                gcm_files['gcm_3'].append(f"    {i+1: <6}")
                gcm_files[  'gcm'].append(f"{b1: <15}    {bins_[i2]: <15}    {i+1: <4}  {i2+1: <4}")
                if i2 == i: gcm_files['gcm_diag'].append(f"    {k: <6}")
        
        for i, l_r_wf in enumerate(bins2copy):
            fld_i = Path( str(i+1) )
            fld_i.mkdir(parents=True, exist_ok=True)
            shutil.copy(l_r_wf[0], fld_i / 'left_wf.bin')
            shutil.copy(l_r_wf[1], fld_i / 'right_wf.bin')
            shutil.copy('taurus_pav.exe', fld_i)
            shutil.copy('input_pav.txt',  fld_i)
            os.chmod(fld_i / 'taurus_pav.exe', 0o777)            
            
        _ = 0
        
    
        # create the scripts
        print("  3. Creating SLURM job scripts.") 
        slurm = _SlurmJob1DPreparation(interaction, len(bins_), valid_J_list,
                                       PAV_input_filename='input_pav.txt',
                                       HWG_input_filename='input_mix.txt')       
        
        for fn_, scr_ in slurm.getScriptsByName().items():
            dst_fld = Path('') if fn_ != 'hw.x' else Path(DEST_FLD_HWG)
            with open(dst_fld / fn_, 'w+') as f: f.write(scr_)
        for fn_, scr_ in gcm_files.items():
            with open(fn_, 'w+') as f: f.write("\n".join(scr_))
        
        # run sbatch
        if RUN_SBATCH: 
            os.system('sbatch sub_1.x')
            print(  "   [Executing] sbatch")
        else: print("   [Not Executing] sbatch")
    
        ## done, go back
        os.chdir(RETURN_FLD)
        print("   * done for z,n=", zn, f' Ks:{k_founds.keys()}, len={len(bins2copy)}\n')
    
    print(f"## Script completed for {MAIN_FLD_TMP} - {interaction}")
    
def oddeven_mix_multiK_from_differentFld_vap(K_list, MAIN_FLD_TMP, interaction, nuclei,
                                        PNP_fomenko=1, Jmax=1, RUN_SBATCH=False):
    """
    Create all the folders with executable, hamil, etc. In this case, using all
    the K states blocked for the nuclei.
        1, 
        2, 
        ,...
        N:[left_wf.bin, right_wf.bin, hamil.* taurus_pav.exe, input_pav.exe]
        
        scripts: 
    
    Args:
    MAIN_FLD_TMP is a template for the placing of the different BU_folder_{inter}_zn:
        
    MULTIK_SAME_FLD: if the folder contains, for each nuclei all the Ks, set True
        otherwise, K is being read from different folders.
        * FALSE
    
    """
    K_list = sorted(K_list)
    print("## Script begin for the list K:", K_list, " ***************** \n")
        
    global input_pav_args 
    input_pav_args[InputTaurusPAV.ArgsEnum.n_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.z_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.j_min]  = min(K_list)
    input_pav_args[InputTaurusPAV.ArgsEnum.j_max]  = max(max(K_list), Jmax)
    
    global input_mix_args
    
    TEMP_BU      = "BU_folder_{}_z{}n{}/"
    DEST_FLD     = 'kmix_PNPAMP_z{}n{}/'
    DEST_FLD_HWG = 'HWG'
    valid_J_list = [i for i in range(1, Jmax+1, 2)]
    
    first_step = True
    nuclei_by_K_found = {}
    path_migration, fld_migration = {}, {}
    # copy pav folders.
    print("  1. Creating folders, Copying binaries, etc for PAV/HWG") 
    for K in K_list:
        print("  K =", K, "  *first_state =", first_step)
        BU_KVAP      =f"{K}_0_VAP/"
        
        MAIN_FLD = Path(MAIN_FLD_TMP.format(K=K))
        if not MAIN_FLD.exists(): 
            print(" [ERROR] Main folder does not exists:", MAIN_FLD)
            return
        
        MAIN_DEST_PATH = MAIN_FLD.parent
        
        if first_step:
            ## if previous k-mix folders clear
            for fld_ in filter(lambda x: os.path.isdir(x)
                                         and x.startswith('kmix_PNPAMP_z'), 
                               os.listdir(MAIN_DEST_PATH)):
                fld_pav = Path(MAIN_DEST_PATH) / fld_
                if fld_pav.exists(): shutil.rmtree(fld_pav)
            
        for z, n in nuclei:
            fld1 = MAIN_FLD / Path(TEMP_BU.format(interaction, z, n))
            
            fld_pav = MAIN_DEST_PATH / DEST_FLD.format(z, n)
            fld_mix = fld_pav / DEST_FLD_HWG
            
            if first_step:                 
                fld_pav.mkdir(parents=True, exist_ok=True)
                fld_mix.mkdir(parents=True, exist_ok=True)
                shutil.copy('taurus_pav.exe', fld_pav)
                shutil.copy('taurus_mix.exe', fld_mix)
                
                pav_obj = InputTaurusPAV(z, n, interaction, **input_pav_args)                
                mix_obj = InputTaurusMIX(z, n, 0, **input_mix_args)
                
                with open(fld_pav / 'input_pav.txt', 'w+') as f:
                    f.write(pav_obj.getText4file())
                with open(fld_mix / 'input_mix.txt', 'w+') as f:
                    f.write(mix_obj.getText4file())
                
                for tail_ in ('.2b', '.com', '.sho'):
                    shutil.copy(fld1 / (interaction+tail_), fld_pav)
            
            
            ## The files migration for each k.
            if not (z,n) in fld_migration:  fld_migration[(z,n)]  = (fld_pav, fld_mix)
            if not (z,n) in path_migration: path_migration[(z,n)] = []
            if not (z,n) in nuclei_by_K_found: nuclei_by_K_found[(z,n)] = {}
            fld_kvap = fld1 / BU_KVAP 
            if not fld_kvap.exists(): print(f" [Error] K={K} folder not present {z},{n}")
            nuclei_by_K_found[(z,n)][K] = fld_kvap.exists()
            
            def_list = filter(lambda x: x.endswith('.bin'), os.listdir(fld_kvap))
            def_list = sort_by_deformation_naming(def_list)
            for bin_ in def_list:
                bin2_ = bin_.replace('.bin', f'_{K}.bin')
                path_migration[(z,n)].append(bin2_)
                shutil.copy(fld_kvap / bin_, fld_pav / bin2_)
        
        if first_step: first_step = False
    
    _ = 0
    # copy pav folders.
    print("  2. Copying binaries, etc for PAV") 
    for zn, k_founds in nuclei_by_K_found.items():
        
        fld_pav, fld_mix = fld_migration[zn]
        bins_ = path_migration[zn]
        
        if not all(k_founds.values()):
            print(f" [ERROR] Non all K were present for z,n={zn}, skipping ,{k_founds}")

        os.chdir(fld_pav)
        _pth_separ = '\\' if os.getcwd().startswith('C:') else '/'
        RETURN_FLD = "/".join([".." for _ in str(fld_pav).split(_pth_separ)])
            
        bins2copy = []
        k, gcm_files = 0, {'gcm': [], 'gcm_diag': [], 'gcm_3': []}
        for i, b1 in enumerate(bins_):
            for i2 in range(i, len(bins_)):
                k += 1
                bins2copy.append( (b1, bins_[i2]) )
                gcm_files['gcm_3'].append(f"    {i+1: <6}")
                gcm_files[  'gcm'].append(f"{b1: <15}    {bins_[i2]: <15}    {i+1: <4}  {i2+1: <4}")
                if i2 == i: gcm_files['gcm_diag'].append(f"    {k: <6}")
        
        for i, l_r_wf in enumerate(bins2copy):
            fld_i = Path( str(i+1) )
            fld_i.mkdir(parents=True, exist_ok=True)
            shutil.copy(l_r_wf[0], fld_i / 'left_wf.bin')
            shutil.copy(l_r_wf[1], fld_i / 'right_wf.bin')
            shutil.copy('taurus_pav.exe', fld_i)
            shutil.copy('input_pav.txt',  fld_i)
            os.chmod(fld_i / 'taurus_pav.exe', 0o777)            
            
        _ = 0
        
    
        # create the scripts
        print("  3. Creating SLURM job scripts.") 
        slurm = _SlurmJob1DPreparation(interaction, len(bins_), valid_J_list,
                                       PAV_input_filename='input_pav.txt',
                                       HWG_input_filename='input_mix.txt')       
        
        for fn_, scr_ in slurm.getScriptsByName().items():
            dst_fld = Path('') if fn_ != 'hw.x' else Path(DEST_FLD_HWG)
            with open(dst_fld / fn_, 'w+') as f: f.write(scr_)
        for fn_, scr_ in gcm_files.items():
            with open(fn_, 'w+') as f: f.write("\n".join(scr_))
        
        # run sbatch
        if RUN_SBATCH: 
            os.system('sbatch sub_1.x')
            print(  "   [Executing] sbatch")
        else: print("   [Not Executing] sbatch")
    
        ## done, go back
        os.chdir(RETURN_FLD)
        print("   * done for z,n=", zn, f' Ks:{k_founds.keys()}, len={len(bins2copy)}\n')
    
    print(f"## Script completed for {MAIN_FLD_TMP} - {interaction}")
    
if __name__ == '__main__':
    
    ## Exists the programs and compile with ifort.
    if False:
        importAndCompile_taurus(use_dens_taurus=False, 
                                pav = not os.path.exists('taurus_pav.exe'), 
                                mix = not os.path.exists('taurus_mix.exe'))
    
    # TESTING_
    # inter  = 'B1_MZ3' 
    # nuclei = [(2, 1), (2, 3)]
    # oddeven_mix_same_K_from_vap(1, 'results', inter, nuclei, 
    #                             PNP_fomenko=7, Jmax=9, )
    
    # raise Exception("STOP-TEST")
    ## 
    inter  = 'B1_MZ4' 
    nuclei = [(12, 11+ 2*i) for i in range(0, 5)]
    # nuclei = [(15, 8+ 2*i)  for i in range(0, 6)]
    # nuclei = [(17, 12+ 2*i) for i in range(0, 5)]
    
    #===========================================================================
    # ## PAV for SINGLE - K
    #===========================================================================
    # X = elementNameByZ[nuclei[0][0]]
    # for K in ( 5,):
    #     MAIN_FLD = f'results/MgK{K}'
    #     oddeven_mix_same_K_from_vap(K, MAIN_FLD, inter, nuclei,
    #                                 PNP_fomenko=7, Jmax=9, RUN_SBATCH=True)
    
    
    #===========================================================================
    # ## PAV - HWG for multi K (K folders separated for each nuclei)
    #===========================================================================
    K_list = [1, 3, 5, 7]
    MAIN_FLD_TMP = 'results/'+elementNameByZ[nuclei[0][0]]+'K{K}'
    oddeven_mix_multiK_from_differentFld_vap(K_list, MAIN_FLD_TMP, inter, nuclei, 
                                             PNP_fomenko=7, Jmax=17, 
                                             RUN_SBATCH=False)
    
    #===========================================================================
    # ## PAV - HWG for multi K (All the K are in each folder)
    #===========================================================================
    # K_list = [1, 3, 5]
    # nuclei = [(9, 8+ 2*i) for i in range(0, 5)]
    # MAIN_FLD = 'DATA_RESULTS/SD_Kblocking_multiK/F_multiK'
    # oddeven_mix_multiK_from_sameFld_vap(K_list, MAIN_FLD, inter, nuclei, 
    #                                     PNP_fomenko=7, Jmax=17, RUN_SBATCH=False)

