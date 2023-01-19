from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.special import sph_harm
from sympy import S
from sympy.physics.quantum.cg import CG

import matplotlib.pyplot as plt
import numpy as np


## Initialize variables
sh_numbers = {}
sh_AntLabel= {}
sp_numbers = {}
sh_from_sp = {}
density_matrix = None
pairing_matrix = None

sh_dim, sp_dim = 0, 0
Rdim, Tdim, Pdim = 0,0,0
integr_method = -1
dim = 0
b_length = None

x, y, z = [], [], []
r, costh, theta, phi = [], [], [], []
map_ang = []
dens_xyz_Z = None
dens_xyz_N = None
pair_rtp_Z = None
pair_rtp_N = None
dens_rtp = None
pair_rtp = None

pair_rtp_Z_list = None  
pair_rtp_N_list = None
dens_xyz_Z_list = None  
dens_xyz_N_list = None
dens_rtp_list = None

weights = []
Z = N = 0


## import the data
def cast_numbers(args, sci_notation=False):
    args_cast = []
    for i, arg in enumerate(args):
        if not '.' in arg:
            args_cast.append(int(arg))
        else:
            if 'D' in arg:
                arg = arg.replace('D', 'E')
            elif i > 2 and sci_notation:
                sign_minus = arg.startswith('-')
                if sign_minus:
                    arg = arg[1:]
                mant, exp_ = arg.split('-')
                arg = ((-1)**sign_minus) * float(mant) * (10**-int(exp_))
            
            args_cast.append(float(arg))
    return args_cast

with open("DIMENS_indexes_and_rhoLRkappas.gut", 'r') as f:
    data = f.read()
    _, block_sh, block_sp, data = data.split('//')
    
    for line in block_sh.split('\n')[1:-1]:
        if len(line) < 5: break
        sh_dim += 1
        # print(f"[{line}]")
        
        line = cast_numbers(line.split())
        i_sh, ant_label, n, l, j  = line[0], line[1], line[2], line[3], line[4]
        sh_numbers[i_sh]  = (n, l, j)
        sh_AntLabel[i_sh] = ant_label
    
    for line in block_sp.split('\n')[1:-1]:
        if len(line) < 5: break
        sp_dim += 1
        # print(f"[{line}]")
        
        line = cast_numbers(line.split())
        i_sp, i_sh, n,l,j  = line[0], line[1], line[2], line[3], line[4]
        m, mt, i_sp_TR     = line[5], line[6], line[7]
        sh_from_sp[i_sp] = i_sh
        sp_numbers[i_sp] = (m, mt, i_sp_TR)
        
    density_matrix = np.zeros([sp_dim, sp_dim])
    pairing_matrix = np.zeros([sp_dim, sp_dim])
    
    for line in data.split('\n')[1:-1]:
        # print(f"[{line}]")
        line = cast_numbers(line.split())
        i, j, dens_re, dens_im = line[0], line[1], line[2], line[3]
        kapp_re, kapp_im       = line[4], line[5]
        if abs(dens_im) > 1.0e-10:
            print("WARNING Imag rho > 0!!", dens_im)
        density_matrix[i-1,j-1] = dens_re
        pairing_matrix[i-1,j-1] = kapp_re
        if i > sp_dim/2 or j > sp_dim/2: 
            N += dens_re
        else:
            Z += dens_re
    Z = np.trace(density_matrix[:sp_dim//2,:sp_dim//2])
    N = np.trace(density_matrix[sp_dim//2:,sp_dim//2:])

# plt.imshow(density_matrix)
# plt.show()

i_ = 0
i_incr = 0 
with open("density_rtp.txt", 'r') as f:
    data = f.readlines()
    for line in data:
        if i_ == 0:
            _, dims = line.split('_')
            dims = cast_numbers(dims.split())
            Rdim, Tdim, Pdim, b_length = dims[0], dims[1], dims[2], dims[3]
            integr_method = dims[4]
            
            r     = np.zeros(Rdim)
            costh = np.zeros(Tdim)
            theta = np.zeros(Tdim)
            phi   = np.zeros(Pdim)
            
            if integr_method == 3:
                dim = Rdim * Tdim
                map_ang = np.zeros([Tdim, Tdim])
            else: 
                dim = Rdim * Tdim * Pdim
                map_ang = np.zeros([Tdim * Pdim, Tdim * Pdim])
            
            x = np.zeros(dim)
            y = np.zeros(dim)
            z = np.zeros(dim)
            dens_xyz_Z_list = np.zeros(dim)
            dens_xyz_N_list = np.zeros(dim)
            dens_rtp_list   = np.zeros(dim)
            pair_rtp_Z_list = np.zeros(dim)
            pair_rtp_N_list = np.zeros(dim)
            
            dens_xyz_N = np.zeros([Rdim, Tdim, Pdim])
            dens_xyz_Z = np.zeros([Rdim, Tdim, Pdim])
            dens_rtp   = np.zeros([Rdim, Tdim, Pdim])
            pair_rtp_Z = np.zeros([Rdim, Tdim, Pdim])
            pair_rtp_N = np.zeros([Rdim, Tdim, Pdim])
            
            weights    = np.zeros(dim)
            
            i_ += 1
            continue
        elif i_ == 1:
            i_ += 1
            continue
        elif len(line) < 5: # last line
            break
        # print(f"[{line}]")
        
        line = cast_numbers(line.split(), sci_notation=True)
        i_r,i_th,i_p  = line[0]-1, line[1]-1, line[2]-1
        r[i_r]      = line[3]
        costh[i_th] = line[4]
        theta[i_th] = np.arccos(line[4])
        phi[i_p]    = line[5]
        
        dens_rtp[i_r, i_th, i_p] = line[6]
        
        dens_rtp_list[i_incr] = line[6]
        weights[i_incr]       = line[8]
        
        i_incr += 1
        i_ += 1

i_ = 0
i_incr = 0
with open('dens_pairing_rtp.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        
        if i_ in (0, 1):
            i_ += 1
            continue
        elif len(line) < 5: # last line
            break
        
        line = cast_numbers(line.split(), sci_notation=True)
        i_r,i_th,i_p = line[0] - 1, line[1] - 1, line[2] - 1
        
        pair_rtp_Z_list[i_incr] = line[6]
        pair_rtp_N_list[i_incr] = line[8]
        
        pair_rtp_Z[i_r, i_th, i_p] = line[6]
        pair_rtp_N[i_r, i_th, i_p] = line[8]
        
        if abs(weights[i_incr] - line[10]) > 1.e-9 :
            print("ERROR ! weigths form XYZ file dont match with spherical RTP")
        
        i_incr += 1
        i_ += 1
        
        # if i_incr > 1000: break

i_ = 0
i_incr = 0
with open("density_xyz.txt", 'r') as f:
    data = f.readlines()
    for line in data:
        
        if i_ in (0, 1):
            i_ += 1
            continue
        elif len(line) < 5: # last line
            break
        
        line = cast_numbers(line.split(), sci_notation=True)
        i_r,i_th,i_p = line[0] - 1, line[1] - 1, line[2] - 1
        # x[i_r-1]     = line[3]
        # y[i_th-1]    = line[4]
        # z[i_p-1]     = line[5]
        x[i_incr]     = line[3]
        y[i_incr]     = line[4]
        z[i_incr]     = line[5]
        
        dens_xyz_Z_list[i_incr] = line[6]
        dens_xyz_N_list[i_incr] = line[8]
        
        dens_xyz_Z[i_r, i_th, i_p] = line[6]
        dens_xyz_N[i_r, i_th, i_p] = line[8]
        
        # print(f"{line}")
        
        if abs(weights[i_incr] - line[10]) > 1.e-9 :
            print("ERROR ! weigths form XYZ file dont match with spherical RTP")
        
        i_incr += 1
        i_ += 1
        
        # if i_incr > 1000: break

#%% Plot Dnsity MAtrix
A = np.matrix.trace(density_matrix)
eigen, _  = np.linalg.eig(density_matrix)
# print(eigen)

plt.figure()
plt.title(f"rhoLR -> tr():  Z={Z:6.4} N={N:6.4} A = {A:6.4}")
plt.imshow(density_matrix, cmap='binary')
plt.colorbar()
plt.xlim([-0.5,39.5])
plt.ylim([39.5,-.5])
plt.show()

#%% PLOT XYZ


#Set colours and render
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')



# ax.set_xlim([-10,10])
# ax.set_ylim([-10,10])
# ax.set_zlim([-10,10])
# #ax.set_box_aspect((1,1,1))
# plt.tight_layout()
# plt.show()


X_cuttoff = 0

#Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Min,Max = -1.5   , 1.5

ax.set_xlim([Min,Max])
ax.set_ylim([Min,Max])
ax.set_zlim([Min,Max])
ax.set_box_aspect((1,1,-1))
ax.scatter(x, y, z, c=dens_xyz_N_list, cmap="bwr", alpha = 0.1, s=dens_xyz_N_list*10000)  # )
#ax.set_cmap("Wistia")

#fig.colorbar(ax, orientation='vertical')
plt.show()

#%% PLOT XYZ density
## ploting one radial density

if integr_method != 3:
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    # plt.title(file_)
    # Make data
    shape = (Tdim, Pdim)
    X1 = np.zeros(shape)
    Y1 = np.zeros(shape)
    Z1 = np.zeros(shape)
    
    for i_r in range(2, Rdim, 5):
        print('i_r', i_r)
        for i_th in range(Tdim):
            for i_p in range(Pdim):
                p, t = phi[i_p], costh[i_th]
                
                dens =  dens_rtp[i_r, i_th, i_p]
                # if quadrature > 1:
                #     dens /= np.exp((r[i_r])**2)
                index_ = i_th, i_p
                
                X1[index_] = dens * np.cos(p) * np.sqrt(1 - t**2)
                Y1[index_] = dens * np.sin(p) * np.sqrt(1 - t**2)
                Z1[index_] = dens * t
            
        ax.plot_surface(X1, Y1, Z1, label=f'r={r[i_r]}', alpha=(1- (i_r/Rdim))*0.35)
    
    plt.show()


#%% PLOT SECTION DENSITY TH/Phi. Comprobaciones numericas angulares
if integr_method != 3:
    file_ = ''
        
    # # Version with slider
    from matplotlib.widgets import Slider
    
    PI_plot = (Pdim)//2 - 1
    
    # Make data.
    def compute_and_plot(ax, phi_contour, Rd, Th):
        ## Contour axis (QuadContourSet ) cannot be updated, need to render again
        # Rd, Th = np.meshgrid(r, theta)
        # Z = np.transpose(density[:, :, phi_contour])
        
        # if quadrature > 1:
        #     x = (r/b_length)**2
        #     Z = np.transpose(np.concatenate(
        #             (density[:, :, phi_contour] / np.exp(x), 
        #              np.flip(density[:, :, phi_contour+PI_plot],axis=1) \
        #                          / np.exp(x)) , 
        #                  axis=1))
        # else:
        Z = np.transpose(np.concatenate(
                (dens_rtp[:, :, phi_contour], 
                  np.flip(dens_rtp[:, :, phi_contour+PI_plot],axis=1)) , 
                      axis=1))
    
        ax.set_title(file_+ f'\n phi = {np.rad2deg(phi[phi_contour]):6.2f} º')
        ax.contourf(Th, Rd, Z)
        # ax.set_thetamin(0)
        # ax.set_thetamax(180)
        ax.set_rmax(8.5)
    
    
    phi_contour = 1
    fig2, axc = plt.subplots()
    plt.subplots_adjust(bottom=0.2) 
    axc = plt.subplot(111, polar=True)
    axc.margins(x=0)
    # axc.set_thetamin(0)
    # axc.set_thetamax(180)
    axc.set_rmax(8.5)
    
    aux_theta = np.arccos(costh) + np.pi
    Rd, Th = np.meshgrid(r, np.concatenate((np.arccos(costh), aux_theta)))
    
    compute_and_plot(axc, phi_contour, Rd, Th)
    axcolor = 'lightgoldenrodyellow'
    axc_fi = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sc_fi = Slider(axc_fi, 'Phi',   0.0, Pdim//2 + 1, valinit=0.0, valstep=1)
    
    def update(val):
        phi_contour_2 = max(int(np.floor(sc_fi.val)-1), 0)
        
        axc.cla()
        compute_and_plot(axc, phi_contour_2, Rd, Th)
        plt.draw()
    
    sc_fi.on_changed(update)
    plt.show()
    


    #%%  DENSITY (R) FOR A SECTION TH/PHI #####################################
    
    print("Begin Slider example")
    i_th_plot, i_phi_plot = len(costh) // 2, 00
    fig3, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25) 
    
    plt.title(file_+ f'  phi = theta = 0 deg')
    steps = Tdim // 5
    
    t = costh[i_th_plot]
    # if quadrature == 2:
    #     x = (r/b_length)**2
    #     dens = np.real(density[:, i_th_plot, i_phi_plot]) / np.exp(x**2)
    # else:
    dens = np.real(dens_rtp[:, i_th_plot, i_phi_plot])
    
    y_, = plt.plot(r, dens, '.-', label=f'th={np.rad2deg(t):6.2f}º')
    
    ax.margins(x=0)
    
    axcolor = 'lightgoldenrodyellow'
    ax_fi = plt.axes([0.25, 0.1, 0.65, 0.03],  facecolor=axcolor)
    ax_th = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    
    # s_fi = Slider(ax_fi, 'Phi', 0.0, 2*np.pi, valinit=0.0)
    # s_th = Slider(ax_th, 'Theta', 0.0,   np.pi, valinit=0.0)
    s_fi = Slider(ax_fi, 'Phi',   0.0, Pdim, valinit=0.0, valstep=1)
    s_th = Slider(ax_th, 'Theta', 0.0, Tdim, valinit=0.0, valstep=1)
    
    def update(val):
        i_phi_plot = max(int(np.floor(s_fi.val)-1), 0)
        i_th_plot  = max(int(np.floor(s_th.val)-1), 0)
        

        y_.set_ydata(np.real(dens_rtp[:, i_th_plot, i_phi_plot]))
        
        # ax.title =  
        ax.set_title(file_+ f'\n phi={np.rad2deg(phi[i_phi_plot]):6.3f} deg  ' + \
                            f' th={np.rad2deg(costh[i_th_plot]):6.3f}')
        fig3.canvas.draw_idle()
    
    
    s_fi.on_changed(update)
    s_th.on_changed(update)
    # for i_th_plot in range(0, Tdim, (Tdim - (Tdim % steps))//steps):
    #     t = theta[i_th_plot]
    #     dens = np.real(density[:, i_th_plot, i_phi_plot])
        
    #     plt.plot(r, dens, '.-', label=f'th={np.rad2deg(t):6.2f}º')
    
    ax.set_ylabel('<dens(r; th,phi)>')
    ax.set_xlabel('r [fm]')
    plt.legend()
    plt.show()

    # #%%  PAIRING TENSOR (R) FOR A SECTION TH/PHI #####################################
    # i_th_plot, i_phi_plot = len(costh) // 2, 00
    # fig4, ax = plt.subplots()
    # plt.subplots_adjust(bottom=0.25) 
    
    # plt.title(file_+ f'  phi = theta = 0 deg')
    # steps = Tdim // 5
    
    # t = costh[i_th_plot]
    # # if quadrature == 2:
    # #     x = (r/b_length)**2
    # #     dens = np.real(density[:, i_th_plot, i_phi_plot]) / np.exp(x**2)
    # # else:
    # kappaZ = np.real(pair_rtp_Z[:, i_th_plot, i_phi_plot])
    # kappaN = np.real(pair_rtp_N[:, i_th_plot, i_phi_plot])
    
    # y_p, = plt.plot(r, kappaZ, 'r.-', label=f'P th={np.rad2deg(t):6.2f}º')
    # y_n, = plt.plot(r, kappaN, 'b.-', label=f'N th={np.rad2deg(t):6.2f}º')
    
    # ax.margins(x=0)
    
    # axcolor = 'lightgoldenrodyellow'
    # ax_fi = plt.axes([0.25, 0.1, 0.65, 0.03],  facecolor=axcolor)
    # ax_th = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    
    # # s_fi = Slider(ax_fi, 'Phi', 0.0, 2*np.pi, valinit=0.0)
    # # s_th = Slider(ax_th, 'Theta', 0.0,   np.pi, valinit=0.0)
    # s_fi = Slider(ax_fi, 'Phi',   0.0, Pdim, valinit=0.0, valstep=1)
    # s_th = Slider(ax_th, 'Theta', 0.0, Tdim, valinit=0.0, valstep=1)
    
    # def updateK(val):
    #     i_phi_plot = max(int(np.floor(s_fi.val)-1), 0)
    #     i_th_plot  = max(int(np.floor(s_th.val)-1), 0)
        
    #     y_p.set_ydata(np.real(pair_rtp_Z[:, i_th_plot, i_phi_plot]))
    #     y_n.set_ydata(np.real(pair_rtp_N[:, i_th_plot, i_phi_plot]))
        
    #     # ax.title =  
    #     ax.set_title(file_+ f'\n phi={np.rad2deg(phi[i_phi_plot]):6.3f} deg  ' + \
    #                         f' th={np.rad2deg(costh[i_th_plot]):6.3f}')
    #     fig4.canvas.draw_idle()
    
    
    # s_fi.on_changed(updateK)
    # s_th.on_changed(updateK)
    # # for i_th_plot in range(0, Tdim, (Tdim - (Tdim % steps))//steps):
    # #     t = theta[i_th_plot]
    # #     dens = np.real(density[:, i_th_plot, i_phi_plot])
        
    # #     plt.plot(r, dens, '.-', label=f'th={np.rad2deg(t):6.2f}º')
    
    # ax.set_ylabel('<kappa(r; th,phi)>')
    # ax.set_xlabel('r [fm]')
    # plt.legend()
    # # plt.show()    






plt.rcParams.update({
  "text.usetex": False
})

#%% profiles of the densty. comaprison
dr = r[1]-r[0]
dth = 0.0399676639999999 #costh[1] - costh[0]
dphi = 0.128228272#phi[1] - phi[0]


b = 1.5
A = 16

cte_ = 16 / ((b**3) * 4 * (np.pi**1.5))
dens_f = lambda u: cte_ * (1 + (2*((u/b)**2))) / np.exp((u/b)**2)
y_SHO = [dens_f(ri) for ri in r]

r0, a = 1.25, 0.524
B  = 0.05
R  = r0 * (A**(1/3))
rho0  =  sum(r * r * (1 + B*(r**2))/ (1 + np.exp((r - R) / a)))*dr
rho0 = 16 / (rho0) # .17 #
y_Fermi   = rho0 * (1 + B*(r**2)) / (1 + np.exp((r - R) / a)) / (4*np.pi)

y_variational = dens_rtp[:,0,0]

plt.figure()
plt.plot(r, y_variational, 'r--',label='HFB *')
plt.plot(r, y_SHO, label='SHO')
plt.plot(r, y_Fermi, label='Fermi_WS')

plt.xlim([0.0, 8.0])
plt.xlabel('r (fm)')
plt.ylabel('Rho')
plt.legend()
plt.show()

print("HF ", sum(r*r*y_variational) * weights[0])
print("WS ", sum(r*r*y_Fermi)  * dr * 4 * np.pi)
print("SHO", sum(r*r*y_SHO)*dr * 4  * np.pi)

#%% Profile of the rearrangement term
# plt.rcParams.update({
#   "text.usetex": True,
#     "font.family": "sans-serif"
# })
# 
# b = 1.5
# rpi = np.pi**0.5
# b3 = b**3
# rOb = r / b

# y_variational = dens_rtp[:,0,0]
# y_alphm1 = map(lambda x: (max(1.e-50, x))**(-2./3), y_variational)
# y_alphm1 = np.array(list(y_alphm1))

# R_00 = (2*2/(rpi *b3)) / np.exp(0.5* (rOb)**2)
# R_01 = (2*4/(3*rpi *b3)) * rOb / np.exp(0.5* (rOb)**2)
# R_02 = (2*8/(15*rpi *b3))* (rOb**2) / np.exp(0.5* (r/b)**2)
# R_03 = (2*16/(105*rpi *b3))* (rOb**3) / np.exp(0.5* (r/b)**2)
# R_10 = (2*4/(3*rpi *b3))* (rOb**0)*(1.5 - (rOb**2)) / np.exp(0.5* (r/b)**2)
# R_11 = (2*8/(15*rpi *b3))* (rOb**1)*(2.5 - (rOb**2)) / np.exp(0.5* (r/b)**2)
# R_12 = (2*16/(105*rpi *b3))* (rOb**2)*(3.5 - (rOb**2)) / np.exp(0.5* (r/b)**2)


# R_functions = (R_00, R_01, R_02, R_03, R_10, R_11, R_12)
# R_funct_str = " R_00, R_01, R_02, R_03, R_10, R_11, R_12".replace(' R_','').split(',')
# R_prod = {}
# for i in range(0,len(R_functions), 1):
#     for j in range(i, len(R_functions), 1):
#         for k in range(j, len(R_functions), 1):
#             R_prod['R_'+R_funct_str[i]+R_funct_str[j]+R_funct_str[k]] = \
#                 R_functions[i] * R_functions[j]*R_functions[k]*(rOb**2)# *y_alphm1

# plt.figure()
# plt.plot(r, np.log10(y_variational), 'k--', label=r'$\rho$')
# plt.plot(r, -np.log10(y_alphm1), 'k', label=r'$\rho^{-2/3}$')
# for lab_, y_123 in R_prod.items():
#     plt.plot(r, np.log10(abs(y_123)))#, label=lab_)
# plt.xlabel('r (fm)')
# plt.legend()
# plt.ylabel(r'$log_{10}\ \rho (r)\\ log_{10}\ (r/b)^2|R_{abcdij}(r)| \\ -log_{10}\ \rho^{\alpha -1}(r)$')
# plt.show()


# plt.figure()
# plt.plot(r, y_variational, 'k--', label=r'$\rho$')
# #plt.plot(r, -np.log10(y_alphm1), 'k', label=r'$\rho^{-2/3}$')
# for lab_, y_123 in R_prod.items():
#     plt.plot(r, y_123*y_alphm1)#, label=lab_)
# plt.xlabel('r (fm)')
# plt.ylabel(r'$Integrand\ of\ \partial\ \overline{v}^{DD} $')
# plt.legend()
# plt.show()


