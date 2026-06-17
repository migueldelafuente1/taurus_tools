'''
Created on 17 jun 2026

@author: delafuente
'''
from scripts1d.projection_scripts import run_diagonal_pavResults_even_even_nuclei

if __name__ == '__main__':
    
    
    Z, N, INTER = 14, 14, 'B1_MZ4'
    FLD_path    = f'BU_folder_{INTER}_z{Z}n{N}'
    
    run_diagonal_pavResults_even_even_nuclei(
        FLD_path, Z, N, INTER,
        default_pav_input = {},
        sort_by_seed_deform_index = True,
        range_2J = (0, 2*7),
        fomenko_points=(7, 7)
    )
    