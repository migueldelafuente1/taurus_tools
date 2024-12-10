'''
Created on 29 nov 2024

@author: delafuente
'''

import os, platform
import shutil
from math import prod
from pathlib import Path

from tools.inputs import InputTaurus, InputTaurusPAV, InputTaurusMIX
from tools.helpers import importAndCompile_taurus, elementNameByZ
from scripts1d.script_helpers import _setUpBatchOrTSPforComputation, \
    RUN_USING_BATCH, _JobLauncherClass
from copy import deepcopy
from organize_pav_hwg_folders import sort_by_deformation_naming

MODE_NOCORE = True

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
    InputTaurusPAV.ArgsEnum.com: 1 if MODE_NOCORE else 0,
    InputTaurusPAV.ArgsEnum.empty_states:   0,
    # InputTaurusPAV.ArgsEnum.cutoff_overlap: 1.0e-9,
}
input_mix_args = {
    InputTaurusMIX.ArgsEnum.opt_convergence_analysis: 1,
    InputTaurusMIX.ArgsEnum.opt_remove_neg_eigen: 1,
    InputTaurusMIX.ArgsEnum.max_energy: 30,
    InputTaurusMIX.ArgsEnum.j_val : 'J_VAL',
    InputTaurusMIX.ArgsEnum.parity: 'P_VAL',
    #
    InputTaurusMIX.CutoffArgsEnum.cutoff_overlap: 1.0e-9,
    InputTaurusMIX.CutoffArgsEnum.cutoff_energy : 0.0,
    InputTaurusMIX.CutoffArgsEnum.cutoff_JzJ2   : 1.0e-6,
    InputTaurusMIX.CutoffArgsEnum.cutoff_ZNA    : 1.0e-6,
    InputTaurusMIX.CutoffArgsEnum.cutoff_norm_eigen: 1.0e-6,
    InputTaurusMIX.CutoffArgsEnum.cutoff_negative_eigen: 1.0e-6
}

#===============================================================================
# SCRIPTS
#===============================================================================    


def sort_by_deformation_naming_nDim(def_list_0):
    """ sorting as [_0.8.bin, _0.6.bin, .. , 0.2.bin, 0.4.bin ...]"""
    def_list_0 = list(def_list_0)
    dim_ = def_list_0[0].split('_').__len__()
    defs_2d  = [(x, x.replace('.bin','').split('_')) for x in def_list_0]
    
    keys_ = [set() for _ in range(dim_)]
    def_dimens = [0, ]*dim_
    for x, args in defs_2d:
        for i in range(dim_):
            keys_[i].add(args[i])
    
    for i in range(dim_):
        def_dimens[i] = len(keys_[i])
        keys_[i] = [(x, float(x)) for x in keys_[i]]
        keys_[i] = sorted(keys_[i], key=lambda x: x[1])
        keys_[i] = [x[0] for x in keys_[i]]
    
    sorted_list = []
    for x, args in defs_2d:
        indx_ = [keys_[i].index(args[i]) for i in range(dim_)]
        indx_ = [k * prod(def_dimens[i+1:]) for i, k in enumerate(indx_)]
        indx_ = sum(indx_)
        sorted_list.append( (indx_, x) )
        
    sorted_list = sorted(sorted_list, key = lambda x: x[0])
    sorted_list = [x[1] for x in sorted_list]
    return sorted_list

def switch_limits_for_Jvals_ee_oe_oo_case(zn, valid_J_list, Jmin):
    """ 
    When having a list of odd-even and even-even nuclei, change the J to the correct one.
    """
    valid_J_list_2 = valid_J_list
    if sum([i%2 for i in zn])% 2 == 0 and Jmin % 2 == 1:
        print("[Waring] non-oe nucleus, changing J_vals for HWG to [even] values.")
        valid_J_list_2 = [i for i in range(0, len(valid_J_list), 2)]
    else:
        if Jmin % 2 == 0:
            print("[Waring] non-ee nucleus, changing J_vals for HWG to [odd] values.")
            valid_J_list_2 = [i for i in range(1, len(valid_J_list), 2)]
    return valid_J_list_2
    

def oe_PAV_diagonal_2dimPNpairing_vap(K_list, PAIR_CONSTRS, MAIN_FLD, 
                                      interaction, nuclei, 
                                      parity = 1, PNP_fomenko=1, Jmin_max=(0, 0), 
                                      RUN_SBATCH=False):
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
    global input_mix_args, _JobLauncherClass, RUN_USING_BATCH
    
    _CALC_NON_AXIAL = True if not K_list else False
    if _CALC_NON_AXIAL: K_list = [0, ]
    MAIN_FLD = Path(MAIN_FLD)
    K_list   = sorted(K_list)
    Jmin, Jmax = Jmin_max
    if Jmin < min(K_list): Jmin = min(K_list)
    
    if isinstance(parity, int):
        if parity == 0: parity = -1
        assert parity in (1, -1), " Put the parities as 1/-1"
        parity = (parity, )
    elif isinstance(parity, (tuple, list)) and len(parity) == 2:
        pass
    else:
        print("Give parity as integer (1,-1) or tuple. EXIT")
        return
    
    assert tuple(parity) in ((1, -1), (-1, 1), (1,), (-1,)), " Put the parities as 1/-1" 
    aux = []
    for par in sorted(parity, reverse=True):
        for K in K_list:
            aux.append((K, par))
    KP_list = deepcopy( aux )
    
    print("## Script begin for the list K, p:", K_list, "\n ***************** \n")    
    
    global input_pav_args 
    input_pav_args[InputTaurusPAV.ArgsEnum.n_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.z_Mphi] = PNP_fomenko
    input_pav_args[InputTaurusPAV.ArgsEnum.j_min]  = Jmin
    input_pav_args[InputTaurusPAV.ArgsEnum.j_max]  = max(max(K_list), Jmax)
    
    PPP = (1 - parity[0]) // 2
    TEMP_BU      = "BU_folder_{}_{}_z{}n{}/"
    DEST_FLD     = 'PNAMP'
    valid_J_list = [i for i in range(Jmin, Jmax+1, 2)]
    
    first_step = True
    nuclei_by_K_found = {}
    path_migration, fld_migration = {}, {}
    # copy pav folders.
    print("  1. Creating folders, Copying binaries, etc for PAV/HWG") 
    for K, par in KP_list:
        
        PPP = (1 - par) // 2
        print("  K, P =",K , PPP, "  *first_state =", first_step)
        BU_KVAP = f"{K}_{PPP}_VAP/" if not _CALC_NON_AXIAL else "PNVAP"
        input_pav_args[InputTaurusPAV.ArgsEnum.parity] = par
        #MAIN_FLD = Path(MAIN_FLD_TMP.format(K=K))
        if not MAIN_FLD.exists(): 
            print(" [ERROR] Main folder does not exists:", MAIN_FLD)
            return
        
        MAIN_DEST_PATH = MAIN_FLD.parent
        
        if first_step:
            ## if previous k-mix folders clear
            for fld_ in filter(lambda x: os.path.isdir(x)
                                         and x.startswith('kmix_PNPAMP'), 
                               os.listdir(MAIN_DEST_PATH)):
                fld_pav = Path(MAIN_DEST_PATH) / fld_
                if fld_pav.exists(): shutil.rmtree(fld_pav)
            
        for z, n in nuclei:
            cnstr = '-'.join([x.replace('_','') for x in sorted(PAIR_CONSTRS)])
            fld_bu = MAIN_FLD / Path(TEMP_BU.format(cnstr, interaction, z, n))
            
            if not os.path.exists(fld_bu):
                print(f"[ERROR] Not found main BU-folder: {fld_bu}")
            ## note, save it in the same BU folder.
            fld_pav = fld_bu  / DEST_FLD 
            fld_migration[(z, n)] = fld_pav
            
            if first_step:                 
                fld_pav.mkdir(parents=True, exist_ok=True)
                shutil.copy('taurus_pav.exe', fld_pav)
                os.chmod(fld_pav / 'taurus_pav.exe', 0o777)
                
                valid_J_list_2 = switch_limits_for_Jvals_ee_oe_oo_case((z,n), valid_J_list, Jmin)
                if valid_J_list != valid_J_list_2:
                    input_pav_args[InputTaurusPAV.ArgsEnum.j_min]  = valid_J_list_2[ 0]
                    input_pav_args[InputTaurusPAV.ArgsEnum.j_max]  = valid_J_list_2[-1]
                
                pav_obj = InputTaurusPAV(z, n, interaction, **input_pav_args)                
                
                with open(fld_pav / 'input_pav.txt', 'w+') as f:
                    f.write(pav_obj.getText4file())
                
                for tail_ in ('.2b', '.com', '.sho', '.01b'):
                    if not (fld_bu / (interaction+tail_)).exists(): continue
                    shutil.copy(fld_bu / (interaction+tail_), fld_pav)
            
            ## The files migration for each k.
            if not (z,n) in path_migration: path_migration[(z,n)] = []
            if not (z,n) in nuclei_by_K_found: nuclei_by_K_found[(z,n)] = {}
            fld_kvap = fld_bu / BU_KVAP 
            if not fld_kvap.exists(): print(f" [Error] K,P ={K},{PPP} folder not present Z,N: {z},{n}")
            nuclei_by_K_found[(z,n)][(K, PPP)] = fld_kvap.exists()
            
            def_list = filter(lambda x: x.endswith('.bin'), os.listdir(fld_kvap))
            def_list = sort_by_deformation_naming_nDim(def_list)
            for bin_ in def_list:
                bin2_ = bin_.replace('.bin', f'_{K}_{PPP}.bin')
                path_migration[(z,n)].append(bin2_)
                shutil.copy(fld_kvap / bin_, fld_pav / bin2_)
        
        if first_step: first_step = False
    
    # copy pav folders.
    print("  2. Copying binaries, etc for PAV") 
    for zn, k_founds in nuclei_by_K_found.items():
        
        fld_pav, bins_ = fld_migration[zn], path_migration[zn]
        
        if not all(k_founds.values()):
            print(f" [ERROR] Non all K,P were present for z,n={zn}, skipping ,{k_founds}")
        
        os.chdir(fld_pav)
        
        RETURN_FLD = "/".join(['..' for _ in fld_pav.parts])
        
        bins2copy = []
        k, gcm_files = 0, {'gcm': [], 'gcm_diag': [], 'gcm_3': []}
        for i, b1 in enumerate(bins_):
            for i2 in range(i, len(bins_)):
                if i2 != i: continue
                k += 1
                bins2copy.append( (b1, bins_[i2]) )
                gcm_files['gcm_3'].append(f"    {k: <6}")
                gcm_files[  'gcm'].append(f"{b1: <15}    {bins_[i2]: <15}    {i+1: <4}  {i2+1: <4}")
                gcm_files['gcm_diag'].append(f"    {k: <6}")
        
        for i, l_r_wf in enumerate(bins2copy):
            fld_i = Path( str(i+1) )
            fld_i.mkdir(parents=True, exist_ok=True)
            shutil.copy(l_r_wf[0], fld_i / 'left_wf.bin')
            shutil.copy(l_r_wf[1], fld_i / 'right_wf.bin')
            shutil.copy('taurus_pav.exe', fld_i)
            shutil.copy('input_pav.txt',  fld_i)
            os.chmod(fld_i / 'taurus_pav.exe', 0o777)            
        
        # create the scripts
        print("  3. Creating SLURM job scripts.", _JobLauncherClass.__name__)
        valid_J_list_2 = switch_limits_for_Jvals_ee_oe_oo_case(zn, valid_J_list, Jmin)
        
        slurm = _JobLauncherClass(interaction, len(bins_), valid_J_list_2,
                                  PAV_input_filename='input_pav.txt',
                                  HWG_input_filename='input_mix.txt',
                                  only_diagonal_PAV=True)     
        
        for fn_, scr_ in slurm.getScriptsByName().items():
            if fn_ == 'hw.x': continue
            dst_fld = Path('')
            with open(dst_fld / fn_, 'w+') as f: f.write(scr_)
            os.chmod(dst_fld / fn_, 0o777)
        for fn_, scr_ in gcm_files.items():
            with open(fn_, 'w+') as f: f.write("\n".join(scr_))
        
        # run sbatch
        if RUN_SBATCH: 
            if RUN_USING_BATCH: 
                os.system('sbatch sub_1.x')
            else:
                os.system('python3 sub_tsp.py')
            print(  f"   [Executing] sbatch/tsp [{RUN_USING_BATCH}]")
        else: print("   [Not Executing] sbatch")
    
        ## done, go back
        os.chdir(RETURN_FLD)
        print("   * done for z,n=", zn, f' Ks:{k_founds.keys()}, len={len(bins2copy)}\n')
    
    print(f"## Script completed for {MAIN_FLD} - {interaction}")

if __name__ == '__main__':
    
    ## Exists the programs and compile with ifort.
    if True:
        importAndCompile_taurus(use_dens_taurus=False, 
                                pav = not os.path.exists('taurus_pav.exe'), 
                                mix = not os.path.exists('taurus_mix.exe'))
    
    _setUpBatchOrTSPforComputation()
    ## 
    inter  = 'B1_MZ5'
    
    #===========================================================================
    # ## Clear the PAV proj-matrix elements folder
    #===========================================================================
    ##   Note 1: Old paths for the K-separated folders
    # flds_ = [f'results/ClK{K}' for K in [1, 3, 5, 7,]]
    # flds_ = [f'results/kmix_PNPAMP_z{z}n{n}' for z,n in nuclei]
    # clear_all_pav_folders_1(flds_, removeProjME=True)
    
    ##   Note 2:
    MAIN_FLD = 'DATA_RESULTS/SD_Kblocking_multiK/' 
    MAIN_FLD = MAIN_FLD + 'Cl/BU_folder_{inter}_z{z}n{n}' # /kmix_PNPAMP
    # MAIN_FLD = 'results/Cl/BU_folder_{inter}_z{z}n{n}'
    # MAIN_FLD = 'results/Cl/BU_folder_{inter}_z{z}n{n}/kmix_PNPAMP'
    
    nuclei = [(17, 10+ 2*i) for i in range(0, 7)]
    flds_  = [MAIN_FLD.format(inter=inter, z=z, n=n) for z,n in nuclei] 
    # flds_ = [f'{FLD_}/kmix_PNPAMP']
    # clear_all_pav_folders(flds_, removeProjME=False)
    # 0/0
    
    #===========================================================================
    # ## PAV for K - Group, only DIAGONAL matrix elements
    #===========================================================================
    K_list = []
    par_   = 1 # ( 1,-1)
    nuclei = {
        #( 8, 8): (5, 0, None),
        #(14,14): (5, 0, None),
        #(10,10): (5, 0, None),
        #(10,11): (5, 0, None),
        #(10,12): (5, 0, None),
        #(10,14): (5, 0, None),
        #(10,16): (5, 0, None),
        #(11,11): (5, 0, None),
        #(12,12): (5, 0, None),
        #(12,13): (5, 0, None),
        #(12,15): (5, 0, None),
        #(13,13): (5, 0, None),
        (14,15): (5, 0, None),
        (16,16): (5, 0, None),
        (16,17): (5, 0, None),
        (18,18): (5, 0, None),
        (18,19): (5, 0, None),
        }
    nuclei = sorted(list(nuclei.keys()))
    kwargs  = {'parity':par_, 'PNP_fomenko':7, 'Jmin_max':(1,11), 'RUN_SBATCH':True}
    MAIN_FLD = 'DATA_RESULTS/SD_Kblocking_multiK/Mg/'
    MAIN_FLD = 'DATA_RESULTS/SD_Odd_pnPairing/HFB_S0MZ5/P_T00_J10/' # testing WIN
    
    MAIN_FLD = 'results/HFB_S0MZ5/P_T00_J10/'
    MAIN_FLD = 'results/HFB_S0MZ5/P_T10_J00/'
    
    ## !! DO NOT CHANGE THE ORDER OF THESE CONSTRAINTS.
    PAIR_CONSTRS = {
        InputTaurus.ConstrEnum.P_T00_J10 ,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T00_J1p1,
        # InputTaurus.ConstrEnum.P_T10_J00 ,
        InputTaurus.ConstrEnum.P_T1m1_J00,
        # InputTaurus.ConstrEnum.P_T1p1_J00,        
    }
    
    oe_PAV_diagonal_2dimPNpairing_vap(K_list, PAIR_CONSTRS, MAIN_FLD, 
                                      inter, nuclei, **kwargs)
    0/0
    