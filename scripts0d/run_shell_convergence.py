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

if not os.getcwd().startswith('C:'):
    sys.path.append(sys.path[0] + '/..')

from tools.Enums import InputParts, Output_Parameters, SHO_Parameters,\
    ValenceSpaceParameters, AttributeArgs, ForceEnum, ForceFromFileParameters,\
    GognyEnum, M3YEnum
from tools.helpers import TBME_SUITE, GITHUB_2BME_HTTP,\
    ValenceSpacesDict_l_ge10_byM, PATH_COM2_IN_2BMESUITE
from tools.hamiltonianMaker import generateCOMFileFromFile
from tools.inputs import InputTaurus
from tools.data import DataTaurus

if not os.path.exists(TBME_SUITE):
    order_ = "git clone {}".format(GITHUB_2BME_HTTP)
    e_ = subprocess.call(order_, shell=True, timeout=180)

#===============================================================================
# 
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
    Z, N = 8, 8
    MZmax_global = 4
    INTERACTIONS = [
        (GognyEnum.B1 , 'B1base_MZ8' , 'B1_MZ{}' ),
        (GognyEnum.D1S, 'D1Sbase_MZ7', 'D1S_MZ{}'),
        (M3YEnum.P2   , 'M3Y_P2base_MZ7', 'M3Y_P2_MZ{}'),
        # (M3YEnum.P6   , 'M3Y_P6base_MZ7', 'M3Y_P6_MZ{}'),
    ]
    params = {
        InputTaurus.ArgsEnum.com  : 1,
        InputTaurus.ArgsEnum.seed : 3,
        InputTaurus.ArgsEnum.iterations: 3000,
        InputTaurus.ArgsEnum.grad_type : 1,
        InputTaurus.ArgsEnum.grad_tol  : 0.001,
        InputTaurus.ArgsEnum.beta_schm : 1, ## 0= q_lm, 1 b_lm, 2 triaxial
        InputTaurus.ArgsEnum.pair_schm : 1,
    }
     
    #===========================================================================
    ## Execution stuff
    for args in INTERACTIONS:
        if os.path.exists(args[0]):
            shutil.rmtree(args[0])
        os.mkdir(args[0])
    
    input = InputTaurus(Z, N, 'hamil', **params)
    
    for MZmax in range(3, MZmax_global +1):
        print(" [ ] Doing MZmax=",MZmax)
        for args in INTERACTIONS:
            interaction, file_base, hamil_file = args
            print(f"      Executing interaction [{interaction}]")
            hamil_file =  hamil_file.format(MZmax)
            __import_and_cut_shellrange_interaction(file_base, hamil_file, MZmax, 
                                                    interaction)
            
            input.interaction = hamil_file
            input.set_inputDDparamsFile(**InputTaurus.getDDParametersByInteraction(interaction))
            __runTaurusBaseSolution(Z, N, hamil_file, input, interaction)
            
            for tl in ('.sho', '.2b', '.com'): 
                shutil.move(f'{hamil_file}{tl}', interaction)
    
    print(" [END] Calculation finished")
    
    ## Plot Stuff
    pass