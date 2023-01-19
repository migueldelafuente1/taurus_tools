# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 20:23:51 2022

@author: Miguel
"""

import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageChops, ImageFont
import io


def colormap_jet(r):
    """ returns a R,G,B tuple for the color for r=[0(blue), 0.5(g), 1(red)] """
    # if isinstance(r, float):
    #     rr = min(r + 0.85, 1)
    rr = 1 / (1 + np.exp(-10*(r - 0.5)))
    # bb = 1 - rr**2
    bb = 1 / (np.exp((10 * (r)**2)))
    gg =  1 / (np.exp((10 * (r - 0.65)**2)))
    return rr, gg, bb

def getData_density_RThetaPhi(file_):
    
    Ntheta, Nphi, Nr = 0, 0, 0
    r,t,p,rho = [], [], [], []
    r_set, t_set, p_set = set(), set(), set()
    
    if os.path.exists(file_):
        print("FIle found:", file_)
        with open(file_, 'r') as f:
            data = f.readlines()
            """                          unprojected          projected   
                r      theta   phi      rho_p     rho_n     rho_p     rho_n
              0.0000  0.1571  0.3142   0.11508   0.11508   0.00000   0.00000
            """
            rho_max =  0.0
            for k, line in enumerate(data[2:]):
                vals = line.split()
                vals = [float(v) for v in vals]
                # print(vals)
                
                r_set.add(vals[0])
                t_set.add(vals[1])
                p_set.add(vals[2])
                
                rho_tot = vals[3]+vals[4]
                rho_max = max(rho_tot, rho_max)
                
                rho.append((vals[3], vals[4], rho_tot))
                
            
            r_set = sorted(list(r_set))
            t_set = sorted(list(t_set))
            p_set = sorted(list(p_set))
            
            Ntheta, Nphi, Nr = len(t_set), len(p_set), len(r_set)
    
    lims = [Nr, Ntheta, Nphi]
    
    return lims, (r, t, p, rho), (r_set, t_set, p_set) 
        

def getData_density_R(file_):
    """ 
    Returns array r and array of density (dens_r(prot), dens(neut), total)
    """
    Nr = 0
    r, rho_r = [], []
        
    if os.path.exists(file_):
        with open(file_, 'r') as f:
            data = f.readlines()
            """              unprojected          projected   
                r      rho_p     rho_n     rho_p     rho_n
              0.000   0.13762   0.14711   0.00000   0.00000
            """
            rho_max =  0.0
            for k, line in enumerate(data[2:]):
                vals = line.split()
                vals = [float(v) for v in vals]
                
                r.append(vals[0])
                rho_tot = vals[1]+vals[2]
                rho_max = max(rho_tot, rho_max)
                
                rho_r.append((vals[1], vals[2], rho_tot))
            Nr = len(r)
    else:
        print("[Error] could not open [{}]".format(file_))
    
    return r, rho_r
        
def _closeThetaPhiSurface(rho, Ntheta, Nphi, Nr, p_set):
    ## this add a phi=phi[0] surface segment at the end of each rho(t,th)
    ## to close the surface
    
    app_ = 0
    for ir in range(Nr):
        for it in range(Ntheta):
            i_tpr_0 = app_ + ((Ntheta*Nphi) * ir) + (Nphi * it)
            i_tpr_f = app_ + ((Ntheta*Nphi) * ir) + (Nphi * it) + Nphi 
            
            vals = rho[i_tpr_0]
            
            rho.insert(i_tpr_f, vals)
            app_ += 1
    
    p_set.append(p_set[0])
    Nphi += 1
    
    return rho, Ntheta, Nphi, Nr, p_set
    
    #%%              PLOTS        %%%%%%%%%%%%%%%%%%%%%%%%

def plotRadial_1D_Density(Nr, Ntheta, Nphi, r_set, th_set, rho):
    ## PLOT OF RADIAL DENSITY
    fig = plt.figure()
    ip = 0 # index of the phi angle to fix
    div_th =  1
    for it in range(0, Ntheta, ceil(Ntheta/div_th)):
        # print("i_theta", it, Ntheta, ceil(Ntheta/div_th))
        y = []
        for ir in range(Nr):
            i_tpr = ((Ntheta*Nphi) * ir) + (Nphi * it) + ip
            # print("i_tpr, ", i_tpr+1)
            try:
                y.append(rho[i_tpr][2])
            except IndexError as e_:
                print(ir,it," unfound k=", i_tpr)
                raise e_
        plt.plot(r_set, y, label=f"th={np.rad2deg(th_set[it]):3.1f} º")
    
    plt.ylabel('dens(r; t,p fixed)')
    plt.xlabel('r [fm]')
    plt.legend()        
    plt.show()


def plot_3D_SurfaceDensity(Nr, Ntheta, Nphi, r_set, th_set, p_set, rho, 
                           animatedGIF=False, gif_title=None):
    ## PLOT OF SURFACES OF EQUAL 
    ## Auxiliar cut section in phi to see the interior surfaces
    rho_max = max([r[2] for r in rho]) ## get the maximum of total density
    cut_facor = 1.0 #0.75 #
    
    if cut_facor > 0.99:
        ## close the surface
        args = _closeThetaPhiSurface(rho, Ntheta, Nphi, Nr, p_set)
        rho, Ntheta, Nphi, Nr, p_set = args
    
    print(f"Nphi: {Nphi}, {len(p_set)}")
    Nphi  = int ( cut_facor * Nphi)
    p_set = p_set[:Nphi]
    print(f"Nphi: {Nphi}, {len(p_set)}")
    
    T, P = np.meshgrid(th_set, p_set)
    R    = 0 * T/(T+0.0001)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d') # add_subplot
    
    dens_cuts = (0.60, )# (0.05, .5, .75) ## % of the maximum density
    # dens_cuts = (0.7 ,)
    axes_c = [None,]*len(dens_cuts)
    for i_cut, dens_cut in enumerate(dens_cuts):
        # process to select the r where the rho probability is in a margin
        rho_top =  dens_cut * rho_max
        for it in range(Ntheta):
            for ip in range(Nphi):
                
                for ir in range(Nr-1, -1, -1):
                    i_tpr = ((Ntheta*Nphi) * ir) + (Nphi * it) + ip
                    # print(it, ip, ir, " k::", i_tpr,"=", 
                    #       r_set[ir], t_set[it], p_set[ip])
                    # print("            rho =", rho[i_tpr][2],"of",rho_top)
                    if rho[i_tpr][2] > rho_top:
                        if ir == 0:
                            R[ip, it] = r_set[ir]
                        print("break", ir)
                        break
                    else:
                        R[ip, it] = r_set[ir]
        
        X = R * np.sin(T) * np.cos(P)
        Y = R * np.sin(T) * np.sin(P)
        Z = R * np.cos(T)

        # Plot the surface.
        axes_c[i_cut] = ax.plot_surface(X, Y, Z, 
                                        cmap=plt.cm.bone,
                                        #color=colormap_jet(dens_cut),
                                        # alpha= 1 - 0.8*(1 - dens_cut)**.5, 
                                        antialiased=True)#cmap=plt.cm.)#YlGnBu_r
    
    # ax.set_zlim(0, 1)
    ax.set_title(f"surfaces constant density = {dens_cuts} * max_rho={rho_max:4.2f}"
                  +"\n (the larger density the less transparent)")
    ax.set_xlabel(r'$x\ [fm]}$')
    ax.set_ylabel(r'$y\ [fm]$')
    ax.set_zlabel(r'$z\ [fm]$')
    
    if not animatedGIF:
        plt.show()
    else: ## export a GIF of the surface rotation
        def fig2img(fig):
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            return img
        
        list_gif= []
        for step in range(180):
            ax.view_init(elev=30., azim=2*float(step))
            
            img = fig2img(fig)
            list_gif.append(img)
        
        if gif_title==None:
            gif_title = 'surface_nucleus.gif' 
        else:
            if not gif_title.endswith('.gif'): gif_title += '.gif'
        
        list_gif[0].save(gif_title, save_all=True, append_images=list_gif[1:],
                         optimize=False, duration=50, loop=0)

# def plot_3D_SurfaceDensity_Animation(Nr, Ntheta, Nphi, r_set, th_set, p_set, rho):
#     ## PLOT OF SURFACES OF EQUAL 
#     ## Auxiliar cut section in phi to see the interior surfaces
    
#     rho_max = max([r[2] for r in rho]) ## get the maximum of total density
#     cut_facor = 1.0 #0.75 #
    
#     if cut_facor > 0.99:
#         ## close the surface
#         args = _closeThetaPhiSurface(rho, Ntheta, Nphi, Nr, p_set)
#         rho, Ntheta, Nphi, Nr, p_set = args
    
#     print(f"Nphi: {Nphi}, {len(p_set)}")
#     Nphi  = int ( cut_facor * Nphi)
#     p_set = p_set[:Nphi]
#     print(f"Nphi: {Nphi}, {len(p_set)}")
    
#     T, P = np.meshgrid(th_set, p_set)
#     R    = 0 * T/(T+0.0001)
    
#     fig = plt.figure()
#     ax = fig.gca(projection='3d') # add_subplot
    
#     dens_cuts = (0.60, )# (0.05, .5, .75) ## % of the maximum density
#     # dens_cuts = (0.7 ,)
#     axes_c = [None,]*len(dens_cuts)
#     for i_cut, dens_cut in enumerate(dens_cuts):
#         # process to select the r where the rho probability is in a margin
#         rho_top =  dens_cut * rho_max
#         for it in range(Ntheta):
#             for ip in range(Nphi):
                
#                 for ir in range(Nr-1, -1, -1):
#                     i_tpr = ((Ntheta*Nphi) * ir) + (Nphi * it) + ip
#                     # print(it, ip, ir, " k::", i_tpr,"=", 
#                     #       r_set[ir], t_set[it], p_set[ip])
#                     # print("            rho =", rho[i_tpr][2],"of",rho_top)
#                     if rho[i_tpr][2] > rho_top:
#                         if ir == 0:
#                             R[ip, it] = r_set[ir]
#                         # print("break", ir)
#                         break
#                     else:
#                         R[ip, it] = r_set[ir]
        
#         X = R * np.sin(T) * np.cos(P)
#         Y = R * np.sin(T) * np.sin(P)
#         Z = R * np.cos(T)

#         # Plot the surface.
#         axes_c[i_cut] = ax.plot_surface(X, Y, Z, 
#                                         cmap=plt.cm.bone,
#                                         #color=colormap_jet(dens_cut),
#                                         # alpha= 1 - 0.8*(1 - dens_cut)**.5, 
#                                         antialiased=True)#cmap=plt.cm.)#YlGnBu_r
    
#     # ax.set_zlim(0, 1)
#     ax.set_title(f"surfaces constant density = {dens_cuts} * max_rho={rho_max:4.2f}"
#                   +"\n (the larger density the less transparent)")
#     ax.set_xlabel(r'$x\ [fm]}$')
#     ax.set_ylabel(r'$y\ [fm]$')
#     ax.set_zlabel(r'$z\ [fm]$')
#     # plt.show()
    
    

#%% main for WINDOWS import data and evaluate the powers of the density for
### elements in the PF shell

def importDensitiesRadialPF(folder_densities):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not folder_densities.endswith('/'): folder_densities += '/'
    
    _coreFile =  'spatial_density_R_40Ca.dat'
    # _coreFile =  'spatial_density_R_16O.dat'
    rCa, rhoCa = getData_density_R(folder_densities +_coreFile)
    ALPHA = 1. / 3
    
    def _testRvaluesMatch(r_core, r_val): #### ******************************
        N, M = len(r_core), len(r_val)
        fail_ = True
        
        if N != M:
            print("[Warning] R dims do not match (C/V):", N, M)
            fail_ *= False
        else:
            v = filter(lambda i: abs(r_core[i]-r_val[i]) > 0.01, range(N))
            v = list(v)
            if len(v)>0:
                fail_ *= False
                print("[Warning] R vals", len(v)," do not match (C/V):")
                for i in v:
                    print('  ', i, r_core[i], r_val[i])
        
        print('[PASS] Test R mathc.')
        return bool(fail_)
        #### ****************************************************************
    
    
    for fil_ in os.listdir(folder_densities):
        
        if fil_.startswith('spatial_density_RThetaPhi'):   continue
        elif not fil_.endswith('.dat') or fil_==_coreFile: continue
                
        zn = fil_.replace('.dat','').split('_')[-1]
        z, n = (int(x) for x in zn[1:].split('n'))
        if z != n: continue
        # if zn not in ('z10n10', "z2n2", "z6n6", "z12n12"):continue
        
        print(fil_)        
        r, dens_ = getData_density_R(folder_densities+fil_)
        match_ = _testRvaluesMatch(rCa, r)
        
        plt.figure()        
        if match_:
            fac = 1 / (z+n)
            d_valn = np.array([d[2] for d in dens_])
            d_core = np.array([d[2] for d in rhoCa])
            d_tot_bench = np.power(d_core + d_valn, ALPHA)
            d_tot_test  = np.power(d_core, ALPHA) + np.power(d_valn, ALPHA)
            d_tot_test2 = np.power(d_core, ALPHA) + fac*np.power(d_valn, ALPHA)
            
            plt.plot(r, np.power(d_core, ALPHA), ':r', label='rho(CO)^a')
            plt.plot(r, np.power(d_valn, ALPHA), ':b', label='rho(VS)^a')
            plt.plot(r, d_tot_bench, '-g', label='[rho(CO) + rho(VS)]^a')
            # plt.plot(r, d_tot_test2, '--m', label='rho(CO)^a + (1/a_vs)rho(VS)^a')
            plt.plot(r, d_tot_test , '--k', label='rho(CO)^a + rho(VS)^a')
        else:
            plt.plot(rCa, [d[2] for d in rhoCa], 'k', label='40Ca') #label='16O') #
            plt.plot(r,   [d[0] for d in dens_], '-.r', label='VS prot')
            plt.plot(r,   [d[1] for d in dens_], '-.b', label='VS neut')
            plt.plot(r,   [d[2] for d in dens_], 'g', label='VS total')
            if match_:
                v = np.array([d[2] for d in dens_])
                c = np.array([d[2] for d in rhoCa])
                plt.plot(r, v+c, 'r', label='CO+VS')
        
        plt.xlabel('r [fm]')
        plt.ylabel('rho(r)')
        plt.title(f" Valence Space Density(r) on PF: {zn}, alpha={ALPHA:5.3}")
        plt.legend()
        plt.show()
        # plt.savefig(folder_densities+f"densalphaPF_{zn}.pdf")
        
# importDensitiesRadialPF('density_rtp_save')


#%% get the densities of the seeds from folder in linux
import os
import subprocess

folder_seeds = 'seedsPFx1/'
template = """Interaction   
-----------
Master name hamil. files      {interaction}
Center-of-mass correction     0
Read reduced hamiltonian      0
No. of MPI proc per H team    0

Particle Number
---------------
Number of active protons      {z}.00
Number of active neutrons     {n}.00  
No. of gauge angles protons   1
No. of gauge angles neutrons  1

Wave Function   
-------------
Type of seed wave function    1 
Number of QP to block         0
No symmetry simplifications   0
Seed random number generation 0
Read/write wf file as text    0
Cutoff occupied s.-p. states  0.00E-00
Include all empty sp states   0
Spatial one-body density      1
Discretization for x/r        100 0.075 
Discretization for y/theta    0   0.00 
Discretization for z/phi      0   0.00 

Iterative Procedure
-------------------
Maximum no. of iterations     0
Step intermediate wf writing  1
More intermediate printing    0
Type of gradient              0
Parameter eta for gradient    0.001E-00
Parameter mu  for gradient    0.001E-00
Tolerance for gradient        0.001E-00

Constraints             
-----------
Force constraint N/Z          1
Constraint beta_lm            1
Pair coupling scheme          0
Tolerance for constraints     1.000E-08
Constraint multipole Q10      0   0.000
Constraint multipole Q11      0   0.000
Constraint multipole Q20      0   0.000
Constraint multipole Q21      1   0.000
Constraint multipole Q22      0   0.000
Constraint multipole Q30      0   0.000
Constraint multipole Q31      0   0.000
Constraint multipole Q32      0   0.000
Constraint multipole Q33      0   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      1   0.000
Constraint multipole Q42      0   0.000
Constraint multipole Q43      0   0.000
Constraint multipole Q44      0   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     0   0.000
Constraint pair P_T00_J1m1    0   0.000
Constraint pair P_T00_J1p1    0   0.000
Constraint pair P_T10_J00     0   0.000
Constraint pair P_T1m1_J00    0   0.000
Constraint pair P_T1p1_J00    0   0.000
Constraint field Delta        0   0.000
"""

file_export = 'spatial_density_R'
interaction = 'D1S_t0_PF'

if os.getcwd().startswith('/home'):
    
    nucleus = []
    print(" SCRIPT extracting the DENSITY R ---------------------------------")
    for fil_ in os.listdir(folder_seeds):
        if not fil_.startswith('final_z'):
            continue
        print("   * Extracting {}.dat from [{}]".format(file_export, fil_))
        
        fil_2 = fil_.replace('final_z', '').replace('.bin', '')
        z, n = fil_2.split('n')
        z, n = int(z), int(n)
        _e = subprocess.call(f'cp {folder_seeds}{fil_} initial_wf.bin', shell=True)
        
        text = template.format(interaction=interaction, z=z, n=n)
        with open('aux_vs_print_dens.INP', 'w+') as f:
            f.write(text)
        
        _e = subprocess.call('./taurus_vap.exe < aux_vs_print_dens.INP > 00.gut', 
                              shell=True,
                              timeout=100)
        
        args = [file_export, f'z{z}n{n}', folder_seeds]
        _e = subprocess.call('cp {0}.dat {2}{0}_{1}.dat'.format(*args), shell=True)
        args[0] = 'DIMENS_indexes_and_rhoLRkappas'
        _e = subprocess.call('cp {0}.gut {2}{0}_{1}.gut'.format(*args), shell=True)
        try :
            _e = subprocess.call('rm *.dat *.red *.gut *.bin', shell=True)
        except BaseException as e:
            pass
    print(" DONE. ")


#%% TEST COLORMAP
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# red = np.linspace(0, 1, 100)
# plt.figure()
#
# r_val,g_val,b_val = colormap_jet(red)
# plt.figure()
# plt.plot(red,r_val,'r')
# plt.plot(red,g_val,'g')
# plt.plot(red,b_val,'b')
# # plt.show()
#
# for r in red:
#     col = (r,0,1-r)
#     col = colormap_jet(r)
#     plt.scatter(r, 0, color=col)
# plt.show()

#%% MAIN


if __name__ == '__main__':
    
    FILE_RTP = 'density_rtp_save/spatial_density_RThetaPhi_40Ca.dat'
    FILE_RTP = 'density_rtp_save/SiliconMZ4/spatial_density_RThetaPhi_34Si.dat'
    # FILE_RTP = 'density_rtp_save/spatial_density_RThetaPhi_z26n32.dat'
    # FILE_R   = 'spatial_density_R.dat'
    FOLDER2PLOT = ''
    
    lims, rtp_rhos_, sets_ = getData_density_RThetaPhi(FILE_RTP)
    plot_3D_SurfaceDensity(*lims, *sets_, rtp_rhos_[-1], animatedGIF=True)
    plotRadial_1D_Density(*lims, sets_[0], sets_[1], rtp_rhos_[-1])
    
    
    
    