'''
Created on 20 nov 2024

@author: delafuente
'''
from tools.Enums import GognyEnum, OutputFileTypes
import shutil, os
import concurrent.futures
import threading
from pathlib import Path

from tools.hamiltonianMaker import TBME_HamiltonianManager
from scripts1d.script_helpers import getInteractionFile4D1S

RESULTS_FLD = ''

thread_local = None

def _changeCWDAndRun(*args):
    
    FLD_header = 'run_2bme_'
    n =      filter(lambda x: x.startswith(FLD_header), os.listdir())
    n = list(filter(lambda x: os.path.isdir(x), n)).__len__()
    TMP_FLD = f'{FLD_header}{n}'
    os.mkdir(TMP_FLD)
    os.chdir(TMP_FLD)
    # move the template xml
    os.mkdir('data_resources')
    shutil.copy('../data_resources/input_D1S.xml', 'data_resources')
    
    print('hello, args=', args)
    # Import and construct 2BHM
    z, n = args[1:3]
    int_args = {(z, n): args[3:6],}
    hamil_surname = args[6]
    kargs = [
        args[7].get('do_Coulomb', True),
        args[7].get('do_LS', True),
        args[7].get('do_BB', True),
    ]
    
    # kargs -> pasarlo directamente
    try:
        inter = getInteractionFile4D1S(int_args, z, n,
                                       do_Coulomb=kargs[0], do_LS=kargs[1], 
                                       do_BB=kargs[2], 
                                       gogny_interaction= args[0])
        global RESULTS_FLD
        # copy results
        for tl_ in OutputFileTypes.members():
            ini_ = f"{inter}{tl_}"
            if os.path.exists(ini_):
                dst = Path('../'+RESULTS_FLD) / f'{inter}{hamil_surname}{tl_}'
                shutil.copy(ini_,  dst)
        os.chdir('..')
    except BaseException as e:
        print("Problem executing the Hamiltonian, returning to main folder")
        os.chdir('..')
    
    # return TMP_FLD, f'{inter}{hamil_surname}'
    
# Custom context manager for threading
class ThreadContextManager:
    def __init__(self, target, *args, **kwargs):
        self.thread = threading.Thread(target=target, args=args, kwargs=kwargs)

    def __enter__(self):
        self.thread.start()  # Start the thread
        return self.thread

    def __exit__(self, exc_type, exc_value, traceback):
        self.thread.join()  # Ensure the thread completes

def run_parallel_2BMatrixElements_Gogny(interaction_args):
    """
    :args interaction_args <dict>
        (z,n): (GOGNY_interaction, MZmax, MZmin, b_length)
    """
    global RESULTS_FLD
    if os.path.exists(RESULTS_FLD):
        shutil.rmtree(RESULTS_FLD)
    os.mkdir(RESULTS_FLD)
    
    global thread_local
    thread_local = threading.local()
    
    TBME_HamiltonianManager.USE_FROM_TAURUS_TOOLS = False
    
    iterable_ = []
    for zn, args in interaction_args.items():
        kwarg = args[4] if len(args) else {}
        args_2 = [args[0], *zn[:2], *args[1:4], zn[2], kwarg]
        # iterable_.append(args_2)

        with ThreadContextManager(_changeCWDAndRun, *args_2):
            print("Thread is running in the background...")
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     executor.map(_changeCWDAndRun, iterable_)
        
    
    print("All running.")
    
    
# def getInteractionFile4D1S(interactions, z,n, do_Coulomb=True, do_LS=True, do_BB=True,
#                            gogny_interaction = GognyEnum.D1S):
    


    
    
if __name__ == '__main__':
    
    ## kwargs, for optional arguments of getInteractionsFile4D1S:
    # do_Coulomb, do_LS, do_BB
    MZmax = 2
    INTERACTIONS_COMPUTE_BY_ZN = {
        (10,10,'')    : (GognyEnum.D1S, MZmax, 0, None,  {}),
        (10,10,'noLS'): (GognyEnum.D1S, MZmax, 0, None, {'do_LS': False,}),
        (10,10,'noBB'): (GognyEnum.D1S, MZmax, 0, None, {'do_BB': False,}),
    }
    
    RESULTS_FLD = 'HAMILS_D1S'
    
    run_parallel_2BMatrixElements_Gogny(INTERACTIONS_COMPUTE_BY_ZN)