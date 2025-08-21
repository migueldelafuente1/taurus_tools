'''
Created on 27 mar 2025

@author: delafuente

Non - Exe based script

    1. From a files with the matrix elements up to one MZ:
        1.1 create the file for the each shell range from MZmax to MZmin
        1.2 Repeat if different interactions.
    2. 

'''
import os, shutil, sys
import subprocess
import xml.etree.ElementTree as et
from pathlib import Path
import numpy as np

if not os.getcwd().startswith('C:'):
    sys.path.append(sys.path[0] + '/..')

from tools.plotter_levels import MATPLOTLIB_INSTALLED
if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

from tools.Enums import InputParts, Output_Parameters, SHO_Parameters,\
    ValenceSpaceParameters, AttributeArgs, ForceEnum, ForceFromFileParameters,\
    GognyEnum, M3YEnum
from tools.helpers import TBME_SUITE, GITHUB_2BME_HTTP,\
    ValenceSpacesDict_l_ge10_byM, PATH_COM2_IN_2BMESUITE, readAntoine
from tools.hamiltonianMaker import generateCOMFileFromFile
from tools.inputs import InputTaurus
from tools.data import DataTaurus, EvolTaurus

if not os.path.exists(TBME_SUITE):
    order_ = "git clone {}".format(GITHUB_2BME_HTTP)
    e_ = subprocess.call(order_, shell=True, timeout=180)

#===============================================================================
## PLOT 
def plotNumberOfShellsConvergences(Z, N, interactions):
    FLD = '20Mg'
    data, data_pn = {}, {}
    evol, evol_pn = {}, {}
    inter_names = [args[0] for args in interactions]
    Ehfb_min = dict([(i[0], 1000) for i in interactions])
    ## Import data
    for args in interactions:
        interaction, _, int_tmplate = args
        data[interaction] = {}
        evol[interaction] = {}
        data_pn[interaction] = {}
        evol_pn[interaction] = {}
        
        if not interaction in os.listdir(FLD):
            continue
        
        out_files = filter(lambda x: x.startswith('out'), 
                           os.listdir(FLD+'/'+interaction))
        for out_ in out_files:
            if interaction.startswith('P'):
                _, zn, _, hamil, mz = out_.replace('.txt', '').split('_')
            else:
                _, zn, hamil, mz = out_.replace('.txt', '').split('_')
            
            data[interaction][mz] = DataTaurus(Z, N, f"{FLD}/{interaction}/{out_}")
            evol[interaction][mz] = EvolTaurus(f"{FLD}/{interaction}/{out_}")
            Ehfb_min[interaction] = min(Ehfb_min[interaction], 
                                        data[interaction][mz].E_HFB)
        ## PN files
        if not interaction+'_pn' in os.listdir(FLD):
            continue
        
        out_files = filter(lambda x: x.startswith('out'), os.listdir(FLD+'/'+interaction+'_pn'))
        for out_ in out_files:
            if interaction.startswith('P'):
                _, zn, _, hamil, mz = out_.replace('.txt', '').split('_')
            else:
                _, zn, hamil, mz = out_.replace('.txt', '').split('_')
            
            data_pn[interaction][mz] = DataTaurus(Z, N, f"{FLD}/{interaction}_pn/{out_}")
            evol_pn[interaction][mz] = EvolTaurus(f"{FLD}/{interaction}_pn/{out_}")
    
    ## Plotting the Energy vs MZ
    _COLOR = 'rbgkmcy'
    _inter = {
        GognyEnum.B1:  ('MZ10', None, ('^', '2')),
        GognyEnum.D1S: ('MZ8', None,  ('o', '.')),
        M3YEnum.P2:    ('MZ8', None, ('s', 'x')),
    }
    fig, ax = plt.subplots(1, 1,)
    for inter, data_mz in data.items():
        args = _inter[inter]
        mz, i_max, _MARKERS = _inter[inter]
        
        x, y   = [], []
        xx, yy = [], []
        #data_mz = []
        data_mz_sorted = [(int(mz[2:]), mz) for mz in data_mz.keys()]
        data_mz_sorted = [ii[1] for ii in sorted(data_mz_sorted)]
        for mz in data_mz_sorted:
            dt = data_mz[mz]
            #if inter in (GognyEnum.D1S, M3YEnum.P2) and mz=='MZ7': continue
            x.append( int(mz[2:]) )
            y.append( dt.E_HFB - Ehfb_min[inter])
        
        if inter in data_pn:
            # for mz, dt in data_pn[inter].items():
            data_mz_sorted = [(int(mz[2:]), mz) for mz in data_pn[inter].keys()]
            data_mz_sorted = [ii[1] for ii in sorted(data_mz_sorted)]
            for mz in data_mz_sorted:
                dt = data_pn[inter][mz]
                if inter in (GognyEnum.D1S, M3YEnum.P2) and mz in ('MZ6', 'MZ7', 'MZ8'): continue
                xx.append( int(mz[2:]) )
                yy.append( dt.E_HFB - Ehfb_min[inter] )
            
        ic = inter_names.index(inter)
        inter_str = inter if (inter != M3YEnum.P2) else f'M3Y-{inter}'
        ax.plot(x, y, label=f"{inter_str}", 
                linestyle='--', marker=_MARKERS[0], color=_COLOR[ic], markerfacecolor='none', markersize=15)
        if yy:
            ax.plot(xx, yy, label=f"{inter_str} (pn)", linestyle='-', marker='.', color=_COLOR[ic], markersize=15 )
    ax.set_xlabel( r'$N\ shells$', fontsize=15)
    ax.set_ylabel( r'$E_{HFB}$',   fontsize=15)
    ax.xaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_tick_params(labelsize=17)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{FLD}/z{Z}n{N}_shellConvergence.pdf")
    
    # plt.show()
    ## Plotting convergence evolution
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ii = -1
    for inter, data_mz in evol.items():
        ii += 1
        I_STEP = 10
        x, y, lab    = [], [], []
        xx, yy, labb = [], [], []
        
        args = _inter[inter]
        mz, i_max, _MARKERS = _inter[inter]
        de = data_mz[mz]
        # for mz, de in data_mz.items():
        lab.append( int(mz[2:]) )
        # x.append( )
        _red_xy = [(i,yi) for i, yi in enumerate(de.e_hfb)]
        _red_xy = filter(lambda iyi: iyi[0]%I_STEP == 0, _red_xy)
        for iyi in _red_xy:
            x.append( iyi[0] )
            y.append( iyi[1] - data[inter][mz].E_HFB + ii*3)
        
        if inter in evol_pn:
            #for mz, de in evol_pn[inter].items():
            # if not mz in evol_pn: mz = 'MZ6'
            dee = evol_pn[inter][mz]
            labb.append( int(mz[2:]) )
            
            step_ = 0 if inter == 'B1' else x[-1]
            if not data_pn[inter][mz].properly_finished: I_STEP = 1
            _red_xy = [(i+step_,yi) for i, yi in enumerate(dee.e_hfb)]
            _red_xy = filter(lambda iyi: iyi[0]%I_STEP == 0, _red_xy)
            for iyi in _red_xy:
                xx.append( iyi[0] )
                yy.append( iyi[1] - data[inter][mz].E_HFB + ii*3)
            
        ic = inter_names.index(inter)
        inter_str = inter if (inter != M3YEnum.P2) else f'M3Y-{inter}'
        mz_int = labb[-1]
        # for i, mz in enumerate(lab):
        kwargs = dict(color=_COLOR[ic], marker=_MARKERS[0])
        ax.plot(x, y, label=f"{inter_str} N={mz_int}", 
                linestyle='--', markerfacecolor='none', 
                **kwargs)
        ax.plot(x[-1], y[-1], markersize=10, markeredgewidth=3,
                **kwargs)
        ax.axhline(y=y[-1], color=_COLOR[ic], linestyle=':')
        if yy:
            #for i, mz in enumerate(labb):
            kwargs = dict(color=_COLOR[ic], marker=_MARKERS[1])
            if i_max: yy = [yy[0][:min(i_max+1, len(yy[0]))], ]
            ax.plot(xx, yy, label=f"{inter_str} N={mz_int} (pn)", linestyle='-', 
                    **kwargs)
            if data_pn[inter][mz].properly_finished:
                ax.plot(xx[-1], yy[-1], markersize=12, markeredgewidth =5,
                        **kwargs)
        
        ax.legend(fontsize=14)
        # ax.set_xlim( (-1, 150) )
        ax.set_ylim( (-3, 15) )
        ax.set_xlabel( r'$Iterations$', fontsize=15)
        ax.set_ylabel( r'$E_{HFB}$',     fontsize=15)
        ax.xaxis.set_tick_params(labelsize=17)
        ax.yaxis.set_tick_params(labelsize=17)
        
        fig.tight_layout()
        fig.savefig(f'{FLD}/z{Z}n{N}_evolution.pdf')
    plt.show()


#===============================================================================


def __runTBMERunnerSuite(xml_filename, file_out, MZmax, sp_list):
    """
    Run process and copy of the files back into the main folder for execution
    """
    os.chdir(TBME_SUITE)
    print(f"    ** [] Running [{TBME_SUITE}] for [{xml_filename}]")
    if os.getcwd().startswith('C:'):
        py3 = 'C:/Users/delafuente/anaconda3/python.exe'
        e_ = subprocess.call(f'{py3} main.py {xml_filename} > temp.txt',
                             timeout= 600, shell=True)
    else: # linux 
        e_ = subprocess.call(f'python3 main.py {xml_filename} > temp.txt',
                             timeout= 600, shell=True)
    print(f"    ** [DONE] Run [{TBME_SUITE}] for [{xml_filename}]: ")
    
    for tail in ('.sho', '.2b'): shutil.copy(f'results/{file_out}{tail}', '..')
    
    ## truncate the COM file
    com_hamil = generateCOMFileFromFile(MZmax, sp_list, file_out, PATH_COM2_IN_2BMESUITE)
    shutil.copy(com_hamil, '..')
    
    os.chdir('..')
    
def __import_and_cut_shellrange_interaction(file_base, hamil_file, MZmax,
                                            interaction=None):
    """ 
    Manage the xml input file for copying from file_base and export to a 
    hamiltoninan named hamil_file, adjusting only the valence space.
    """
    global L_MAX
    for tail in ('.sho', '.2b', '.com'):
        if tail in file_base:
            print(f"Do not put the extension of the interaction file in 'file_base'")
            return
        if not file_base + tail in os.listdir():
            print( f"Not found [{file_base}{tail}]" )
            return
        shutil.copy(file_base+tail, TBME_SUITE)
    
    with open(file_base + '.sho', 'r') as f:
        hbaromega = f.readlines()[4].split()[1]
    
    ## XML File preparation
    path_xml     = '../data_resources/template.xml'
    xml_filename = 'input_filename.xml'
    interaction = interaction if interaction else file_base.replace('.2b', '')
    _TT = '\n\t\t'
    
    tree = et.parse(path_xml)
    root = tree.getroot()
    out_    = root.find(InputParts.Output_Parameters)
    outfn_  = out_.find(Output_Parameters.Output_Filename)
    outfn_.text = hamil_file
    aux_tit = f"Processed interaction [{interaction}]: MZ={MZmax}"
    title_  = root.find(InputParts.Interaction_Title)
    title_.text = aux_tit
    
    sho_ = root.find(InputParts.SHO_Parameters)
    hbo_ = sho_.find(SHO_Parameters.hbar_omega)
    hbo_.text = hbaromega
    
    val_ = root.find(InputParts.Valence_Space)
    sp_list = []
    for MZ in range(MZmax +1):
        for qn in ValenceSpacesDict_l_ge10_byM[MZ]:
            if readAntoine(qn, l_ge_10=True)[1] > L_MAX: continue
            
            _ = et.SubElement(val_, ValenceSpaceParameters.Q_Number, 
                              attrib={AttributeArgs.ValenceSpaceArgs.sp_state: qn,
                                      AttributeArgs.ValenceSpaceArgs.sp_energy:''})
            _.tail = _TT
            sp_list.append(qn)
        
    forces = root.find(InputParts.Force_Parameters)
    f2  = et.SubElement(forces, ForceEnum.Force_From_File,  
                        attrib={AttributeArgs.ForceArgs.active : 'True'})
    f2.text = _TT
    _ = et.SubElement(f2, ForceFromFileParameters.file, 
                      attrib={AttributeArgs.name : file_base+'.2b'})
    _.tail=_TT
    _ = et.SubElement(f2, ForceFromFileParameters.options,
                      attrib={AttributeArgs.FileReader.ignorelines : '1',
                              AttributeArgs.FileReader.constant:     '1',
                              AttributeArgs.FileReader.l_ge_10:      'True'})
    _.tail='\n\t'
    f2.tail = '\n\t'
    
    tree.write(xml_filename)
    
    ## TBME execution
    shutil.copy(xml_filename, TBME_SUITE)
    __runTBMERunnerSuite(xml_filename, hamil_file, MZmax, sp_list)
    
    
def __runTaurusBaseSolution(Z,N, hamil_name, input_taurus: InputTaurus, fld_2_save):
    """
    run the program.
    """
    INP = input_taurus.DEFAULT_INPUT_FILENAME
    with open(INP, 'w+') as f:
        f.write(input_taurus.getText4file())
    with open(input_taurus.INPUT_DD_FILENAME, 'w+') as f:
        f.write(input_taurus.get_inputDDparamsFile())
    
    os.system(f'./taurus_vap.exe < {INP} > out.txt')
    if os.getcwd().startswith('C:'):
        for f in ('out.txt', 'final_wf.bin', 'eigenbasis_h.dat'):
            with open(f, 'w+') as ff: ff.write("TESTFILE\n...")
    
    shutil.copy('out.txt', f"{fld_2_save}/out_z{Z}n{N}_{hamil_name}.txt")
    shutil.copy('final_wf.bin', f"{fld_2_save}/seed_z{Z}n{N}_{hamil_name}.bin")
    
    for dat in DataTaurus.DatFileExportEnum.members():
        if os.path.exists(f"{dat}.dat"):
            shutil.copy(f"{dat}.dat", f"{fld_2_save}/{dat}_z{Z}n{N}_{hamil_name}.dat")
        
    

    
#===============================================================================
# 
#===============================================================================


if __name__ == '__main__':
    
    #===========================================================================
    Z, N = 12, 8
    MZmax_global = 10
    INTERACTIONS = [
        (GognyEnum.B1 , 'B1base_MZ10' , 'B1_MZ{}' ),
        (GognyEnum.D1S, 'D1Sbase_MZ8', 'D1S_MZ{}'),
        # (GognyEnum.D1S, 'D1Sbase_MZ7', 'D1Sdd_MZ{}'),
        (M3YEnum.P2   , 'M3Y_P2base_MZ8', 'M3Y_P2_MZ{}'),
        # (M3YEnum.P6   , 'M3Y_P6base_MZ7', 'M3Y_P6_MZ{}'),
    ]
    params = {
        InputTaurus.ArgsEnum.com  : 1,
        InputTaurus.ArgsEnum.seed : 3, # 0 
        InputTaurus.ArgsEnum.iterations: 700,
        InputTaurus.ArgsEnum.grad_type : 1,
        InputTaurus.ArgsEnum.grad_tol  : 0.001,
        InputTaurus.ArgsEnum.beta_schm : 1, ## 0= q_lm, 1 b_lm, 2 triaxial
        InputTaurus.ArgsEnum.pair_schm : 1,
    }
    params_dd = {
        InputTaurus.InpDDEnum.r_dim :    16,
        InputTaurus.InpDDEnum.omega_dim: 14,
    }
    
    L_MAX = 5
    #===========================================================================
    ## Execution stuff
    if not os.getcwd().startswith('C'):
        for args in INTERACTIONS:
            if os.path.exists(args[0]):
                shutil.rmtree(args[0])
            os.mkdir(args[0])
        
        input = InputTaurus(Z, N, 'hamil', **params)
        
        for MZmax in range(3, 4):#MZmax_global +1):
            print(" [ ] Doing MZmax=",MZmax)
            for args in INTERACTIONS:
                if params[InputTaurus.ArgsEnum.seed] in (0, 4) and MZmax > 6: 
                    print("   ** Continue, will diverge!!")
                    continue
                
                interaction, file_base, hamil_file = args
                print(f"      Executing interaction [{interaction}]")
                hamil_file =  hamil_file.format(MZmax)
                __import_and_cut_shellrange_interaction(file_base, hamil_file, MZmax, 
                                                        interaction)
                
                input.interaction = hamil_file
                input.set_inputDDparamsFile(**InputTaurus.getDDParametersByInteraction(interaction),
                                            **params_dd)
                __runTaurusBaseSolution(Z, N, hamil_file, input, interaction)
                
                for tl in ('.sho', '.2b', '.com'): 
                    shutil.move(f'{hamil_file}{tl}', interaction)
    
        print(" [END] Calculation finished")
    
    
    ## Plot Stuff
    else:
        plotNumberOfShellsConvergences(Z, N, INTERACTIONS)