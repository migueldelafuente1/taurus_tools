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

#===============================================================================
# SCRIPTS MANAGER
#===============================================================================

class _SlurmJob1DPreparation():
    
    """
        This auxiliary class complete the script templates to keep along with 
    the 1-Dimension wf- PAV calculation:
        SUB_1: group job for the paralelization
        JOB_1: unit job for SUB_1
        CAT  : preparation to extract all the resultant projected matrix elements
        HWG  : sub-job to iterate the HWG calculation by J
    
    Usage:
        Giving all necessary arguments, the instance saves the script texts in
        the attributes:
            job_1
            sub_1
            script_cat
            script_hwg
    """
    
    _TEMPLATE_SLURM_JOB = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-ARRAY_JOBS_LENGTH
#
## max = N*(N+1)/2 being N the number of q-states (prompt from preparegcm.f)

export OMP_NUM_THREADS=1
ulimit -s unlimited
var=$SLURM_ARRAY_TASK_ID

full_path=$PWD"/"
cd $full_path
cd $var

workdir=$PWD
mkdir -p /scratch/delafuen/
mkdir -p /scratch/delafuen/$SLURM_JOB_ID
chmod 777 PROGRAM
cp -r  PROGRAM INPUT_FILE left_wf.bin right_wf.bin /scratch/delafuen/$SLURM_JOB_ID
cp -r  ../HAMIL.* /scratch/delafuen/$SLURM_JOB_ID

cd  /scratch/delafuen/$SLURM_JOB_ID

./PROGRAM < INPUT_FILE > OUT

mv /scratch/delafuen/$SLURM_JOB_ID/OUT $workdir/
#mv /scratch/delafuen/$SLURM_JOB_ID/*.me $workdir/
mv /scratch/delafuen/$SLURM_JOB_ID/*.bin $workdir/
rm /scratch/delafuen/$SLURM_JOB_ID/*
rmdir /scratch/delafuen/$SLURM_JOB_ID/
"""
    
    _TEMPLATE_SLURM_SUB = """#!/bin/bash
## max = N*(N+1)/2 being N the number of q-states (prompt from preparegcm.f)

tt="1-23:59:59"
rang="1-ARRAY_JOBS_LENGTH"

sbatch --output=/dev/null --array $rang --time $tt  $PWD/job_1.x
"""
    
    _TEMPLATE_CAT_ME_STATES = """#!/bin/bash
rm projmatelem_states.bin
mkdir outputs_PAV
rm outputs_PAV/* 

for var in {1..ARRAY_JOBS_LENGTH}; do
var3=$(cat gcm_3 | head -$var | tail -1 | awk '{print$1}')
cat $var3"/projmatelem_states.bin" >> projmatelem_states.bin

cp $var3"/OUT" outputs_PAV"/OUT_"$var3
done

cp gcm* outputs_PAV"""
    
    _TEMPLATE_CAT_ME_STATES_PYTHON = """import shutil, os

OUT_fld  = 'outputs_PAV'
pme_files = [
    'projmatelem_states.bin', 
    'projmatelem_occnumb.bin', 'projmatelem_M1.bin', 'projmatelem_M2.bin',
    'projmatelem_E1.bin',      'projmatelem_E2.bin', 'projmatelem_E3.bin'
]
if os.path.exists(OUT_fld):  shutil.rmtree(OUT_fld)
for pme_file in pme_files: if os.path.exists(pme_file]):  os.remove(pme_file)
os.mkdir(OUT_fld)

def copy_stuff(dir_list):
    for fld_ in dir_list:
        fld_ = fld_.strip()
        if os.path.exists(fld_):
            for pme_file in pme_files:
                if pme_file in os.listdir(fld_): 
                    os.system("cat {}/{} >> {}".format(fld_, pme_file, pme_file))
                else: print(" [ERROR] not found for {}".format(fld_))
                if 'OUT' in os.listdir(fld_):
                    shutil.copy("{}/OUT".format(fld_), 
                                "{}/OUT_{}".format(OUT_fld, fld_ ))
                else: print("     [ERROR 2] not found OUT for {}".format(fld_))
    print("* done for all files")

if os.path.exists('gcm_3'):  # gcm_file
    with open('gcm_3', 'r') as f:
        dir_list = f.readlines()
        dir_list = [fld_.strip() for fld_ in dir_list]
        print(dir_list)
        copy_stuff(dir_list)
else: # without gcm_file
    dir_list = filter(lambda x: os.path.isdir(x) and x.isdigit(), os.listdir())
    dir_list = sorted([int(x) for x in dir_list])
    dir_list = [str(x) for x in dir_list]
    print(dir_list)
    copy_stuff(dir_list)"""
    
    _TEMPLATE_JOB_HWX = """#!/bin/bash

ulimit -s unlimited 

LIST_JVALS
chmod 777 PROGRAM
for var in $LIST; do
sed s/"J_VAL"/$var/ INPUT_FILE > INP0
./PROGRAM < INP0 > $var".dat"
done
"""
    
    _TEMPLATE_PREPARE_PNPAMP = """#!/bin/bash
## max = N*(N+1)/2 being N the number of q-states (prompt from preparegcm.f)

for var in {1..ARRAY_JOBS_LENGTH}; do
var1=$(cat gcm | head -$var | tail -1 | awk '{print$1}')
var2=$(cat gcm | head -$var | tail -1 | awk '{print$2}')
var3=$(cat gcm | head -$var | tail -1 | awk '{print$3}')
var4=$(cat gcm | head -$var | tail -1 | awk '{print$4}')

cp $var1 left_wf.bin
cp $var2 right_wf.bin
mkdir $var
cp PROGRAM INPUT_FILE left_wf.bin right_wf.bin $var
cd $var
chmod 777 PROGRAM
cd ../
done"""
    
    
    TAURUS_PAV = 'taurus_pav.exe'
    TAURUS_HWG = 'taurus_mix.exe'
    
    class ArgsEnum:
        JOBS_LENGTH = 'ARRAY_JOBS_LENGTH'
        INPUT_FILE = 'INPUT_FILE'
        PROGRAM    = 'PROGRAM'
        LIST_JVALS = 'LIST_JVALS'
        HAMIL = 'HAMIL'
    
    def __init__(self, interaction, number_of_wf, valid_J_list, 
                 PAV_input_filename='', HWG_input_filename=''):
        """
        Getting all the jobs for
        """
        self.hamil = interaction
        self.jobs_length = str(number_of_wf * (number_of_wf + 1) // 2)
        if (HWG_input_filename == ''):
            HWG_input_filename = self.ArgsEnum.INP_hwg
        if (PAV_input_filename == ''):
            PAV_input_filename = self.ArgsEnum.INP_pav
        
        
        ## JOB-PARALLEL
        self._prepare_job_and_submit(PAV_input_filename)
        
        ## PREPARE PNAMP
        self.prepare_pnpamp = self._TEMPLATE_PREPARE_PNPAMP
        self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.JOBS_LENGTH,
                                                          self.jobs_length)
        # self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.HAMIL,
        #                                                   self.hamil)
        self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.INPUT_FILE,
                                                          PAV_input_filename)
        self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.PROGRAM, 
                                                          self.TAURUS_PAV)
        ## CAT
        self.script_cat = self._TEMPLATE_CAT_ME_STATES
        self.script_cat = self.script_cat.replace(self.ArgsEnum.JOBS_LENGTH,
                                                  self.jobs_length)
        
        ## HWG
        J_vals = " ".join([str(j) for j in valid_J_list])
        J_vals = f'LIST="{J_vals}"'
        self.script_hwg = self._TEMPLATE_JOB_HWX
        self.script_hwg = self.script_hwg.replace(self.ArgsEnum.LIST_JVALS,
                                                  J_vals)
        self.script_hwg = self.script_hwg.replace(self.ArgsEnum.INPUT_FILE,
                                                  HWG_input_filename)
        self.script_hwg = self.script_hwg.replace(self.ArgsEnum.PROGRAM,
                                                  self.TAURUS_HWG)
    
    ##
    def _prepare_job_and_submit(self, PAV_input_filename):
        
        self.job_1 = self._TEMPLATE_SLURM_JOB
        self.job_1 = self.job_1.replace(self.ArgsEnum.JOBS_LENGTH,
                                        self.jobs_length)
        self.job_1 = self.job_1.replace(self.ArgsEnum.HAMIL, self.hamil)
        self.job_1 = self.job_1.replace(self.ArgsEnum.INPUT_FILE, 
                                        PAV_input_filename)
        self.job_1 = self.job_1.replace(self.ArgsEnum.PROGRAM, self.TAURUS_PAV)
        
        self.sub_1 = self._TEMPLATE_SLURM_SUB
        self.sub_1 = self.sub_1.replace(self.ArgsEnum.JOBS_LENGTH,
                                        self.jobs_length)
    
    def getScriptsByName(self):
        
        scripts_ = {
            'sub_1.x': self.sub_1, 
            'job_1.x': self.job_1,
            'hw.x': self.script_hwg,
            'cat_states.me.x': self.script_cat,
            'cat_states.py'  : self._TEMPLATE_CAT_ME_STATES_PYTHON,
            'run_pnamp.x': self.prepare_pnpamp,
        }
        
        return scripts_


class _TaskSpoolerJob1DPreparation(_SlurmJob1DPreparation):
    
    """
        This class prepare the task spooler job to run if no SLURM is present
        The job scripts are not directly runnable, requires previous setting 
        before running.
    """
    
    _TEMPLATE_SLURM_JOB = """##
## Job to be run, using: job_tsp.py [working_folder]
##

from sys import argv
from datetime import datetime
import os, shutil

hamil      = 'HAMIL'
input_file = 'INPUT_FILE'
program    = 'PROGRAM'
LOG_JOB_FN = 'submitted_tspError.LOG'

try:
    terminal_args_given = argv
    assert len(terminal_args_given) == 2, \
        "invalid argument given, 2 required, got: {}".format(terminal_args_given)
    
    _, fld_ = terminal_args_given
    
    hamil_files = filter(lambda x: x.startswith(hamil), os.listdir())
    for hamil_f in hamil_files:
        shutil.copy(hamil_f, fld_)
    
    os.chdir (fld_)
    assert input_file in os.listdir(), \
        "Not found input arg[{}] in folder[{}]".format(input_file, fld_)
    
    os.system('./{} < {} > OUT'.format(program, input_file))
    
    ## Clear the hamiltonians from  folder
    for hamil_f in hamil_files: os.remove(hamil_f)
    os.chdir ('..')
    
    print("  # done folder {}".format(fld_))
    
except BaseException as e:
    with open(LOG_JOB_FN, 'w+') as f:
        # Convert to string
        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("[ERROR] [{}] fld:[{}]: {}", str(dt_str, fld_, str(e)))
"""
    
    _TEMPLATE_SLURM_SUB = """##
## Folders numerated 1, 2, 3 ... with the wavefunctions, taurus and input 
## already inside them

import os, shutil

JOB_NAME = 'job_tsp.py'

pav_list = filter(lambda x: x.isdigit(), os.listdir())
pav_list = sorted(list(map(lambda x: int(x), pav_list)))

print("SUBMIT_JOBs [START]")
for fld_ in pav_list:
    fld_ = str(fld_)
    
    # change to the folder and run with tsp
    os.system("tsp python3 {} {}".format(JOB_NAME, fld_))

## Getting logs and clear 
os.system('tsp -C')
print("SUBMIT_JOBs [DONE]")"""
    
    def _prepare_job_and_submit(self, PAV_input_filename):
        
        self.job_1 = self._TEMPLATE_SLURM_JOB
        self.job_1 = self.job_1.replace(self.ArgsEnum.HAMIL, self.hamil)
        self.job_1 = self.job_1.replace(self.ArgsEnum.INPUT_FILE, 
                                        PAV_input_filename)
        self.job_1 = self.job_1.replace(self.ArgsEnum.PROGRAM, self.TAURUS_PAV)
        
        self.sub_1 = self._TEMPLATE_SLURM_SUB
    
    def getScriptsByName(self):
        
        scripts_ = {
            'sub_tsp.py': self.sub_1, 
            'job_tsp.py': self.job_1,
            'hw.x': self.script_hwg,
            'cat_states.me.x': self.script_cat,
            'cat_states.py'  : self._TEMPLATE_CAT_ME_STATES_PYTHON,
            'run_pnamp.x': self.prepare_pnpamp,
        }
        
        return scripts_

#===============================================================================
# 
#===============================================================================

_JobLauncherClass = _SlurmJob1DPreparation
RUN_USING_BATCH   = True

def _setUpBatchOrTSPforComputation():
    """
    In case of running in Linux system- define if is TSP or SLURM system present
    """
    global RUN_USING_BATCH, _JobLauncherClass
    if not os.getcwd().startswith('C:'):
        os.system('which sbatch > HASBATCH')
        with open('HASBATCH', 'r') as f:
            aux = f.read()
            print("  [SET UP]: which sbatch:\n", aux, '\n')
            if aux == '' or 'not sbatch' in aux: 
                _JobLauncherClass = _TaskSpoolerJob1DPreparation
                RUN_USING_BATCH   = False
        os.remove('HASBATCH')
    print("  [SET UP] RUN_USING_BATCH =", RUN_USING_BATCH)
