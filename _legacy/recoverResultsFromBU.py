# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:26:07 2022

@author: Miguel

Import the results from a folder, get their DataTaurus information, then short
for the other exe results 2D plots to plot.
"""

import os
import shutil
import subprocess
import zipfile
from _legacy.exe_isotopeChain_taurus import DataTaurus
import matplotlib.pyplot as plt

# FOLDER_IMPORT = 'PN_mixingD1STests/USDB/isovector_zips'
FOLDER_IMPORT = 'PN_mixingD1STests/USDB/isoscalar_zips'
FOLDER_OUTPUT = 'PN_mixingD1STests/USDB'

#%% Import the zips
zip_files = {}

for i, zfilename in enumerate(os.listdir(FOLDER_IMPORT)):
    print(" %%% zipfile=", zfilename)
    if not zfilename.endswith('.zip'):
        continue
    # if i == 0: continue
    # if i > 1: 
    #     break ## TODO: Remove after testing
    
    z, n = zfilename.split('_')[1].split('-')[0][1:].split('n')
    z, n = int(z), int(n)
    PAIR_C = '_'.join(zfilename.split('_')[2:5]).split('-')[-1]
    
    zip_path = FOLDER_IMPORT + '/' + zfilename
    dir_2_extract = FOLDER_OUTPUT + '/' + zfilename.replace('.zip','')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_2_extract)
    
    zip_files[zfilename] = []
    
    #%%  get the output textfiles
    results_taurus = {}
    order_bp = []
    for f_out in os.listdir(dir_2_extract+'/BU_results'):
        if f_out.endswith('.bin'):
            continue
        path_out = dir_2_extract + '/BU_results/' + f_out
        res = DataTaurus(z,n,path_out)
        
        if not res.proton_numb: 
            continue
        
        zip_files[zfilename].append(res)
        
        b2, pp = res.b20_isoscalar,  getattr(res, PAIR_C)
        aux_ind = 10000*(b2+0.1) + pp
        order_bp.append((aux_ind, len(order_bp), b2, pp))
    
    # sort
    order_bp = sorted(order_bp, key=lambda x: x[0], reverse=False)
    i, j = 0, -1
    prev_b2 = None
    key_order = []
    for _, ii, b2, pp in order_bp:
        
        ## convert the b-p values to index from (0, 0:N)
        if prev_b2 == None:
            prev_b2 = b2
        if abs(b2 - prev_b2) < 0.005:
            j += 1
        else:
            i += 1
            j = 0
            prev_b2 = b2
        
        print(b2, pp, " -> ", i, j)
        
        if (i,j) in results_taurus:
            print( " IJ index already in:", i,j)
        else:
            results_taurus[(i, j)] = zip_files[zfilename][ii]
            key_order.append((i, j))
    
    #%% export the data in a format readable for plotter
    txt_ = ''
    for index_ in key_order:
        txt_ += "{} = {}\n".format(index_, 
                                   results_taurus[index_].getAttributesDictLike)
    # 'export_PBTESz18n18_D1S_PT1p1J00'
    PAIR_C = PAIR_C.replace('_', '')
    export_fn = f'export_PBTESz{z}n{n}_D1S_{PAIR_C}.txt'
    with open(os.path.join(FOLDER_OUTPUT, export_fn), 'w+') as f:
        f.write(txt_)
    
    #%% rm the folder for zip
    # os.rmdir(dir_2_extract)
    shutil.rmtree(dir_2_extract, ignore_errors=True)
    