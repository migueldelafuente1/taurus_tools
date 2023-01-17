'''
Created on Jan 10, 2023

@author: Miguel
'''
from tools.executors import ExeTaurus1D_DeformQ20, ExecutionException
from tools.inputs import InputTaurus



def run_q20_surface(nucleus, q_min=-10, q_max=10, N_max=50):
    
    interaction = "hamil"
    
    for z, n in nucleus:
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.r_dim : 12,
               InputTaurus.InpDDEnum.omega_dim : 14})
        
        input_args_start = {
            InputTaurus.ArgsEnum.com : 1,
            InputTaurus.ArgsEnum.seed: 5,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 0, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
        }
                
        try:
            exe_ = ExeTaurus1D_DeformQ20(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min,  q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution()
            exe_.run()
            # exe_.tearDown()
        except ExecutionException as e:
            print(e)


if __name__ == '__main__':
    
    nucleus = [(2, 4), (4, 4)]
    run_q20_surface(nucleus, q_min=-5, q_max=5, N_max=10)