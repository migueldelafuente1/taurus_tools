'''
Created on 13 sept 2023

@author: delafuente
'''
from tools.data import DataTaurusMIX

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except Exception:
    MATPLOTLIB_INSTALLED = False
    print("WARNING :: Matplotlib not installed. Do not evaluate plot modules.")

import os
from copy import copy, deepcopy
import collections

_BASE_CLS_ATTRS = {
    '_horizontal_margin' : -0.07,
    '_vertical_margin'   : -0.07,
    'title_size'    : 0.1,
    'subtitle_size' : 0.05,
    'LINE_SIZE'     : 10,
    'DECIMALS_ENER' : 3,
    'FONTSIZE_SUBTITLES' : 'x-large',
    'line_box_size'         : 0.6,
    'energy_value_size'     : 0.35,
    'j_pi_label_size'       : 0.15,
    'neck_horizontal_size'  : 0.2, ## of the level_line box
    }

class _BaseLevelGraph(object):
    
    """
    Abstract class for the development of a single level scheme, de
    """
    
    _horizontal_margin = -0.07
    _vertical_margin   = -0.07
    
    title_size = 0.1
    subtitle_size = 0.05
    
    LINE_SIZE     = 10
    DECIMALS_ENER = 3
    FONTSIZE_SUBTITLES = 'x-large'
    
    line_box_size     = 0.6
    energy_value_size = 0.35
    j_pi_label_size   = 0.15
    
    neck_horizontal_size = 0.2 ## of the level_line box
    
    @classmethod
    def resetClassAttributes(cls):
        for attr_, default_ in _BASE_CLS_ATTRS.items():
            setattr(cls, attr_, default_)
        
    
    def __init__(self, title='', *args, **kwargs):
        
        self._data_energy = []     # Fundamental (absolute) energies
        self._data_exc_energy = [] # Save the relative energies
        self._data_angMom = []
        self._data_parity = []
        self._graph_x = []       # coordinate x for the level segment for each E
        self._graph_y = []       # coordinate y for the level segment for each E
        
        self._scopeJbypar = {'+': [], '-': []}
        self.Emin = None
        self.Emax = None
        
        self._coords_subtitle = []
        self.title = 'BTitle'
        
        if title:
            self.title = str(title)
    
    @property 
    def n_levels(self):
        return len(self._data_energy)
    
    def setData(self):
        """ 
        Process to filter or get the data for the levels (Parity, J, excit, ...)
        """
        raise BaseException("Abstract Method, implement me!")
    
    def render_box(self, pyplot_obj, print_excit_energies=False):
        """ Define what to plot and matplotlib orders (legends, grids ...)"""
        raise BaseException("Abstract Method, implement me!")
    
    def _update_Jpar_scope(self):
        """  Update the range of states 
        example :: par = { +:[0, 2 ,4 ,6], -:[1, 3, 5]}"""
        
        for i, j in enumerate(self._data_angMom):
            par = self._data_parity[i]
            if not j in self._scopeJbypar[par]:
                self._scopeJbypar[par].append(j)
                self._scopeJbypar[par] = sorted(self._scopeJbypar[par])
            

class EnergyLevelGraph(_BaseLevelGraph):
    
    """
    Graphs for Energy vs J and parity.
    example:
       6+ ____ 5.16
       
       4+ ____ 3.66
       6+ ____ 2.36
       
       0+ ____ 0.00
    
    """
    
    def setData(self, levels_data, program=None):
        
        ## if program == 'taurus_hwg':      select methods of processing
        
        ## Reset data if called.
        self._data_angMom = []   ## The vlaues are in 2J, remember when print
        self._data_energy = []
        self._graph_x = []
        self._graph_y = []
        self._coords_subtitle = ()
        
        for line in levels_data.split('\n'):
            if len(line) == 0: continue
            
            args = line.strip().split()
            
            if '/' in args[0]: 
                args[0] = args[0].split('/')[0]
                self._data_angMom.append(int(args[0]))
            else:
                self._data_angMom.append(2 * int(args[0]))
            self._data_parity.append(args[1])
            self._data_energy.append(float(args[3]))
        
        self.Emin = min(self._data_energy)
        self.Emax = max(self._data_energy)
        
        for i, ener in enumerate(self._data_energy):
            self._data_exc_energy.append(ener - self.Emin)
        
        ## sort by energy (required for stacking the labels)
        for i in range(len(self._data_energy)-1):
            
            for j in range(len(self._data_energy) - i -1):
                
                e_j = copy(self._data_energy[j])
                j_j = copy(self._data_angMom[j])
                p_j = copy(self._data_parity[j])
                eej = copy(self._data_exc_energy[j])
                
                e_i = copy(self._data_energy[j+1])
                j_i = copy(self._data_angMom[j+1])
                p_i = copy(self._data_parity[j+1])
                eei = copy(self._data_exc_energy[j+1])
                
                if e_j > e_i:
                    self._data_energy[j+1] = e_j
                    self._data_angMom[j+1] = j_j
                    self._data_parity[j+1] = p_j
                    self._data_exc_energy[j+1] = eej
                    
                    self._data_energy[j] = e_i
                    self._data_angMom[j] = j_i
                    self._data_parity[j] = p_i
                    self._data_exc_energy[j] = eei
        
        self._update_Jpar_scope()
    
    def render_box(self, plt_ax, print_excit_energies=False):
        """
        Takes an pyplot_axes object and append the values for the data
        """
        if print_excit_energies:
            ener_list_selected = self._data_exc_energy
        else:
            ener_list_selected = self._data_energy
        
        format_e = " {:>+6.%df}" % self.DECIMALS_ENER
        if print_excit_energies:
            format_e = " {:>8.%df}" % self.DECIMALS_ENER
            
        for i, ener_i in enumerate(ener_list_selected):
            
            x1, x2, x3, x4 = self._graph_x[i]
            y1, y2, y3, y4 = self._graph_y[i]
            
            en_str = format_e.format(ener_i)
            
            if self._data_angMom[i] % 2 == 0:
                jp_str = " {:2}{:1} ".format(self._data_angMom[i] // 2, 
                                             self._data_parity[i])
            else:
                jp_str = " {:2}/2{:1} ".format(self._data_angMom[i], 
                                               self._data_parity[i])
            plt_ax.annotate(jp_str, xy=(x1, y1),
                            horizontalalignment='right')
            plt_ax.plot(self._graph_x[i], self._graph_y[i], 'k')
            plt_ax.annotate(en_str, xy=(x4, y4))
        
        #print the title beneath
        title_str = self.title
        if print_excit_energies:
            title_str += "\nE0({:+5.3f}) MeV".format(self._data_energy[0])
        plt_ax.annotate(title_str, xy=self._coords_subtitle, 
                        fontsize=self.FONTSIZE_SUBTITLES,
                        horizontalalignment='center')
        
    
    def add_graph_x(self, x_tuple):
        assert len(x_tuple) == 4, "Invalid x tuple"
        self._graph_x.append([*x_tuple, ])
    
    def add_graph_y(self, y_tuple):
        assert len(y_tuple) == 4, "Invalid y tuple"
        self._graph_y.append([*y_tuple, ])



class EnergyByJGraph(EnergyLevelGraph):
    
    """ 
    Import and organize the results by energy, J and excitation collections
    """
    def __init__(self, title='', *args, **kwargs):
        super(EnergyByJGraph, self).__init__(title=title, *args, **kwargs)
        
        self._surface_bypar : dict = None
        
    def setData(self, levels_data, program=None):
        
        ## Import and sort the data
        EnergyLevelGraph.setData(self, levels_data, program=program)
        
        ## Organize by J and P
        surface_bypar= {'+': [], '-': [],}
        
        for i, e_i in enumerate(self._data_energy):
            j_i = self._data_angMom[i]
            p_i = self._data_parity[i]
            element_ = (j_i, e_i)
            
            if surface_bypar[p_i].__len__() == 0:
                surface_bypar[p_i].append( [element_, ] )
            else:
                ## Append to a previous excitation collection, or create one.
                found_, skip_ = False, False
                for jj in range(len(surface_bypar[p_i])):
                    if skip_: continue
                    
                    if j_i in list(zip(*surface_bypar[p_i][jj]))[0]:
                        found_ = True
                    else:
                        surface_bypar[p_i][jj].append( element_ )
                        skip_ = True      ## jj might not be the last collection 
                
                if found_ and (not skip_):
                    surface_bypar[p_i].append( [element_ , ] )
        
        for p in '+-':
            if len(surface_bypar[p]) == 0:
                del surface_bypar[p]
        self._surface_bypar = surface_bypar
        self._update_Jpar_scope()
    

    def render_box(self, plt_ax, color=''):
        
        _LS_MK = 'o*vx+d^>s<hDPX'*2
        for par, values_ in self._surface_bypar.items():
            for i, exc_col in enumerate(values_):
                # NOTE:: This is to align the branch from left to right
                exc_col = dict(exc_col)
                exc_col = collections.OrderedDict(sorted(exc_col.items()))
                exc_col = [(x,y) for x, y  in exc_col.items()]
                
                x, y = list(zip(*exc_col))
                x = [xx/2 for xx in x]   # J is stored as 2J
                
                if i == 0:
                    label_ = '{}: ({}){}={:4.2f} MeV'.format(self.title, i, par,
                                                            self.Emin)
                else:
                    label_ = '{}: ({}){}'.format(self.title, i, par)
                    
                
                plt_ax.plot(x, y, '{}{}'.format(color, _LS_MK[i]),
                            linestyle='dashed' if i>0 else '-',
                            label = label_)
                # plt_ax.annotate(en_str, xy=(x4, y4))
        
        #print the title beneath
        title_str = self.title
        # plt_ax.annotate(title_str, xy=self._coords_subtitle, 
        #                 fontsize=self.FONTSIZE_SUBTITLES,
        #                 horizontalalignment='center')
        
    def add_graph_x(self, x_tuple):
        raise Exception("This method cannot be used for this class .")
    
    def add_graph_y(self, y_tuple):
        # self._graph_y.append(*y_tuple )
        raise Exception("This method cannot be used for this class .")
        
class TransitionalLevelGraph(_BaseLevelGraph):
    pass


#===============================================================================
# IMAGE GENERATOR AND MANAGEING
#===============================================================================

class BaseLevelContainer(object):
    '''
    Combine one or more Level Objects, to be defined after the construction
    '''
    RELATIVE_PLOT  = False  ## Plots the difference, with respect to the min.
    VERT_TEXT_CONV = None   ## Parameter to define the proper stack E value boxes
    E_BOX_UNIT_HEIGHT = None
    
    ONLY_PAIRED_STATES = False  ## Only show groups of states linked for the sigma
    MAX_NUM_OF_SIGMAS  = 99     ## Limit the number of sigma-excitations to show. 

    def __init__(self):
        '''
        Constructor
        '''
        self.number_of_levels = 0
        self._levelGraphs = []
        self._levelData   = []
        self._levelTitles = []
        self._fundamental_energies = [] # to set the 0 for the relative plotting
        
        self._maxEner = -999999.9
        self._minEner =  999999.9
        
        self._maxHeight = self._maxEner
        self._minHeight = self._minEner
        
        self._LX_box  = 10.0 # (arbitrary)
        self._LXmax   = self._LX_box 
        
        self.global_title = ''
        self._fig  = None
        
    def add_LevelGraph(self, level_object: _BaseLevelGraph, title = ''):
        """
        Introduce a _BaseLevelGraph derived object, already with its data.
        The method implements rules to check, gather information and store.
        
        :level_object is a data-level element to be render
        :title will set a Title label bellow or above the level column
        """
        
        self._levelGraphs.append(level_object)
        self._levelTitles.append(title)
        
        self._minEner = min(self._minEner, level_object.Emin)
        self._maxEner = max(self._maxEner, level_object.Emax)
        
        ## TODO: register the maximum / minimum energies
        fund_ener = level_object.Emin
        self._fundamental_energies.append(fund_ener)
        
        if len(self._levelGraphs) > 1:
            self._LXmax += self._LX_box
        

    def plot(self,export_filename=False, figaspect=None):
        
        ## TODO: set the top and lower bounds of X and Y, 
        ## (in case of overlaping levels)
        if len(self._levelGraphs) > 2:
            Nlev = len(self._levelGraphs)
            _BaseLevelGraph._horizontal_margin *= 0.7 * Nlev
            # _BaseLevelGraph.neck_horizontal_size /= 0.5 * Nlev
            _BaseLevelGraph.DECIMALS_ENER = 2 if Nlev < 4 else 1
            _BaseLevelGraph.FONTSIZE_SUBTITLES = 'large'
            
        self._setLevelBoundsAndCoordinates()
        
        self._renderLevelGraphs(figaspect=figaspect)
        
        if export_filename:
            export_filename.replace(".pdf", "")
            self._fig.savefig(export_filename + '.pdf')
    
    def _filterStatesForThisLevelObject(self, lev_obj : _BaseLevelGraph):
        """
        Conditions to count and append states in the plot
        """
        indx_groups = {'+': [], '-': []}
        indx_j_stored = []
        
        for par1 in ('+', '-'):
            for jj in lev_obj._scopeJbypar[par1]:
                k = 0
                for i, j in enumerate(lev_obj._data_angMom):
                    par = lev_obj._data_parity[i]
                    if (j!=jj) or (i in indx_j_stored) or (par!=par1): continue
                    
                    indx_j_stored.append(i)
                    if ((j == lev_obj._scopeJbypar[par1][0]) 
                        or (k >= len(indx_groups[par]))):
                        indx_groups[par].append([])
                    indx_groups[par][k].append(i)
                    k += 1
        ## FILTERS for the number and set of E(J) curves
        count_sigmas_par = {'+': 0, '-': 0}
        sigmas_to_rm = []
        for par, sigma_indx in indx_groups.items():
            
            for i in range(len(sigma_indx)):
                if self.ONLY_PAIRED_STATES:
                    if len(sigma_indx[i]) != len(lev_obj._scopeJbypar[par]):
                        sigmas_to_rm += sigma_indx[i]
                        continue
                
                count_sigmas_par[par] += 1  
                if count_sigmas_par[par] > self.MAX_NUM_OF_SIGMAS:
                    sigmas_to_rm += sigma_indx[i]
            
        for i in sorted(sigmas_to_rm, reverse=True):
            _ = lev_obj._data_angMom.pop(i)
            _ = lev_obj._data_energy.pop(i)
            _ = lev_obj._data_parity.pop(i)
            _ = lev_obj._data_exc_energy.pop(i)
    
    def _get_y_stack_and_bounds(self, lev_obj, y_min, y_max):
        
        j_states = set(lev_obj._data_angMom)
        j_states_count = [lev_obj._data_angMom.count(j) for j in j_states]
        count_j = dict([(j, 0) for j in j_states])
        
        if self.RELATIVE_PLOT:
            if self.MAX_NUM_OF_SIGMAS > max(j_states_count):
                y_max = max(y_max, lev_obj.Emax - lev_obj.Emin)
                y_min = min(y_min, 0.0)
            else:
                for i, e_ in enumerate(lev_obj._data_energy):
                    count_j[lev_obj._data_angMom[i]] += 1
                    y_max = max(y_max, e_ - lev_obj.Emin)
                    y_min = min(y_min, 0.0)
                    if all([lj >= self.MAX_NUM_OF_SIGMAS for lj in count_j.values()]): 
                        break
                
            y_stack = lev_obj._data_exc_energy[0]
            list_lev_select = lev_obj._data_exc_energy
                
        else:
            if self.MAX_NUM_OF_SIGMAS > max(j_states_count):
                y_max = max(y_max, lev_obj.Emax)
                y_min = min(y_min, lev_obj.Emin)
            else:
                for i, e_ in enumerate(lev_obj._data_energy):
                    count_j[lev_obj._data_angMom[i]] += 1
                    y_max = max(y_max, e_)
                    y_min = min(y_min, e_)
                    if all([lj >= self.MAX_NUM_OF_SIGMAS for lj in count_j.values()]): 
                        break
            y_stack = lev_obj.Emin
            list_lev_select = lev_obj._data_energy
        
        return y_min, y_max, y_stack, list_lev_select
    
    def _setLevelBoundsAndCoordinates(self):
        """
        From the levels in the data, calculate the box size to put the energy
        and J-p value considering the stacking.
        Also the global tops for the margins
        """
        y_max = self._maxHeight
        y_min = self._minHeight
        
        ## conversion fontsize = 10 / MeV range
        if not self.VERT_TEXT_CONV:
            if not self.E_BOX_UNIT_HEIGHT:
                n = min([lev_obj.n_levels for lev_obj in self._levelGraphs])
                self.E_BOX_UNIT_HEIGHT = min(0.03, 
                                             (self._maxEner - self._minEner)/n)
            
            self.VERT_TEXT_CONV = (self._maxEner - self._minEner)\
                                         * self.E_BOX_UNIT_HEIGHT    #0.05
        
        lev_obj : _BaseLevelGraph = None
        
        for i, lev_obj in enumerate(self._levelGraphs):
            self._filterStatesForThisLevelObject(lev_obj)
            
            lev_box_x = self._LX_box * (1 + 2*lev_obj._horizontal_margin)
            # NOTE: (horizontal margin is negative)
            lev_e_x = lev_box_x * (1 - lev_obj.j_pi_label_size
                                     - lev_obj.energy_value_size)
            # Reference for the box
            x_b = (i + lev_obj._horizontal_margin) * self._LX_box
            
            args = self._get_y_stack_and_bounds(lev_obj, y_min, y_max)
            y_min, y_max, y_stack, list_lev_select = args
                
            for indx, ener_i in enumerate(list_lev_select):
                
                # Segment points: x1-x2 (left neck) x2-x3 (h line) x3-x5 (r.neck)
                x1 = x_b + lev_obj.j_pi_label_size * lev_box_x
                x4 = x1  + lev_e_x
                x2 = x1  + lev_obj.neck_horizontal_size * lev_e_x
                x3 = x4  - lev_obj.neck_horizontal_size * lev_e_x
                
                lev_obj.add_graph_x([x1, x2, x3, x4])
                # TODO: do the stack for the y values 
                
                if indx > 0:
                    y_stack += self.VERT_TEXT_CONV
                    if ener_i > y_stack:
                        y_stack = ener_i
                
                lev_obj.add_graph_y([y_stack, ener_i, ener_i, y_stack] )
                # Update the y max for the stack
                y_max   = max(y_max, y_stack)
            
            #self._levelGraphs[i] = lev_obj
            
        # Final global range of the y AXES
        self._maxHeight = y_max
        self._minHeight = y_min
        
        y_range = abs(y_max - y_min)
        
        for i, lev_obj in enumerate(self._levelGraphs):
            x_sub = (lev_obj._graph_x[0][3] + lev_obj._graph_x[0][0]) / 2
            y_sub = self._minHeight + 2 * y_range * lev_obj._vertical_margin
            lev_obj._coords_subtitle = (x_sub, y_sub) # x1, loower y
            
            #self._levelGraphs[i] = lev_obj
        
    
    def _renderLevelGraphs(self, figaspect=(6.4, 4.8)):
        ax_ : plt.Axes = None
        _fig, ax_ = plt.subplots(figsize=figaspect)
        
        ## TODO: Iterate for render all the level positions
        level_obj : _BaseLevelGraph = None
        for indx_, level_obj in enumerate(self._levelGraphs):
            level_obj.render_box(ax_, print_excit_energies=self.RELATIVE_PLOT)
        
        # Remove axis bars and x labels
        ax_.spines['right'] .set_visible(False)
        ax_.spines['top']   .set_visible(False)
        ax_.spines['left']  .set_visible(False)
        ax_.spines['bottom'].set_visible(False)
        
        # plt.xticks(visible=False)
        ax_.set_xticklabels([])
        ax_.set_xticks([])
        
        # Set tops and units for the x axis
        y_range = 2 * abs(self._maxHeight - self._minHeight)
        ax_.set_ylim(self._minHeight + y_range * level_obj._vertical_margin, 
                     self._maxHeight - y_range * level_obj._vertical_margin)
        
        x_lims  = ax_.get_xlim()
        x_range = (x_lims[1] - x_lims[0]) / len(self._levelGraphs)
        x_range *= 2 * self._levelGraphs[0]._horizontal_margin
        ax_.set_xlim(x_lims[0] + x_range,  x_lims[1])
        
        ax_.set_title(self.global_title, fontdict={'fontfamily' : 'sans-serif',
                                                   'fontsize' : 20,})
        ax_.set_ylabel('Energy (MeV)')
        if self.RELATIVE_PLOT:
            plt.ylabel('Excitation Energy (MeV)')
        
        self._fig = _fig
        plt.show()
        
#===============================================================================
# MAIN (examples)
#===============================================================================



class JLevelContainer(BaseLevelContainer):
    
    """ 
    Plot to present the levels by the J in the x axis, (representing rotational 
    bands).
    """
    
    def _setLevelBoundsAndCoordinates(self):
        
        """
        From the levels in the data, calculate the box size to put the energy
        and J-p value considering the stacking.
        Also the global tops for the margins
        """
        y_max = self._maxHeight
        y_min = self._minHeight
        
        ## conversion fontsize = 10 / MeV range
        if not self.VERT_TEXT_CONV:
            self.VERT_TEXT_CONV = (self._maxEner - self._minEner) * 0.05
        
        lev_obj : EnergyByJGraph = None
        
        if self.RELATIVE_PLOT:
            y_max = self._maxEner - self._minEner
            y_min = 0.0
        else:
            for i, lev_obj in enumerate(self._levelGraphs):
                y_max = max(y_max, lev_obj.Emax)
                y_min = min(y_min, lev_obj.Emin)
        
        self._maxHeight = y_max
        self._minHeight = y_min
        
        ## Arange
        if not self.RELATIVE_PLOT: 
            return 
        
        for i, lev_obj in enumerate(self._levelGraphs):
            
            count_sigmas_par = {'+': 0, '-': 0}
            for par, jener in lev_obj._surface_bypar.items():
                
                sigmas_to_rm = []
                for i in range(len(jener)):
                    
                    if not self.RELATIVE_PLOT:
                        new_ = [(vls[0], vls[1] - self._minEner) for vls in jener[i]]
                    else:
                        new_ = [(vls[0], vls[1] - lev_obj.Emin) for vls in jener[i]]
                    
                    ## FILTERS for the number and set of E(J) curves
                    if self.ONLY_PAIRED_STATES:
                        if len(jener[i]) != len(lev_obj._scopeJbypar[par]):
                            sigmas_to_rm.append(i)
                            continue
                    count_sigmas_par[par] += 1  
                    if count_sigmas_par[par] > self.MAX_NUM_OF_SIGMAS:
                        sigmas_to_rm.append(i)
                        continue
                    
                    lev_obj._surface_bypar[par][i] = deepcopy(new_)
                
                for i in sorted(sigmas_to_rm, reverse=True):
                    _ = lev_obj._surface_bypar[par].pop(i)
                
    
    def _renderLevelGraphs(self, figaspect= plt.rcParams["figure.figsize"]):
        
        ax_ : plt.Axes = plt.subplot()
        
        ## TODO: Iterate for render all the level positions
        level_obj : _BaseLevelGraph = None
        COLORS_ = 'rbgkcmpy'
        x_range = 0
        for indx_, level_obj in enumerate(self._levelGraphs):
            level_obj.render_box(ax_, color=COLORS_[indx_])
            
            x_r = max(level_obj._data_angMom) - min(level_obj._data_angMom)
            x_range = max(x_r, x_range) if indx_>0 else x_r
        
        ax_.tick_params(axis="y",direction="in")
        ax_.tick_params(axis="x",direction="in")
        
        
        # Set tops and units for the x axis
        
        
        plt.title(self.global_title, fontdict={'fontfamily' : 'sans-serif',
                                               'fontsize' : 20,})
        plt.ylabel('Energy (MeV)')
        plt.xlabel('J')
        plt.legend()
        if self.RELATIVE_PLOT:
            plt.ylabel('Excitation Energy (MeV)')
            ax_.annotate(r"Emin={:9.3f} MeV".format(self._minEner), 
                         xy=(0.75 * x_range, - 0.075 * self._maxHeight), 
                         fontsize=level_obj.FONTSIZE_SUBTITLES,
                         horizontalalignment='center')
            ax_.set_ylim([self._minHeight - 0.1 * self._maxHeight, 
                          self._maxHeight + 1])
        
        #plt.show()

def getAllLevelsAsString(folder_path):
    """
    Get the levels in the HWG folder to 
    """
    str_ = ''
    if isinstance(folder_path, str): folder_path = [folder_path, ]
    
    for folder_ in folder_path:
        if not os.path.exists(folder_): 
            print(" [ERROR] Couldn't find the folder for HWG:", folder_)
            continue
        for file_ in os.listdir(folder_):
            if not file_.endswith('.dat'): continue
            print("  .. importing (hwg) file:", file_)
            dat_ = DataTaurusMIX(folder_+file_)
            str_ += dat_.getSpectrumLines()
    return str_

if __name__ == '__main__':
    
    example_levels = """
    0  +   1   -221.910    0.000    0.000    0.000   2.8069   2.7941   2.8005   2.9161    1.000000   12.000001   11.999951   23.999952   0.00017   0.00876
    0  +   2   -209.044   12.866    0.000    0.000   2.8455   2.8343   2.8399   2.9532    1.000000   11.999999   12.000015   24.000015   0.00008   0.02530
    2  +   1   -219.895    0.000    0.000    0.000   2.8141   2.8013   2.8077   2.9230    1.000000   12.000004   12.000013   24.000017   2.00000   0.00358
    2  +   2   -208.327   11.568    0.000    0.000   2.8528   2.8417   2.8473   2.9603    1.000000   11.999995   12.000120   24.000115   2.00002   0.00905
    4  +   1   -215.242    0.000    0.000    0.000   2.8196   2.8068   2.8132   2.9283    1.000000   12.000002   12.000005   24.000007   4.00001   0.00308
    4  +   2   -204.904   10.338    0.000    0.000   2.8438   2.8327   2.8383   2.9516    1.000000   11.999998   11.999959   23.999957   3.99997   0.00687
    6  +   1   -207.897    0.000    0.000    0.000   2.8382   2.8254   2.8318   2.9462    1.000000   12.000001   12.000003   24.000004   6.00001   0.00320
    6  +   2   -199.266    8.631    0.000    0.000   2.8223   2.8108   2.8166   2.9309    1.000000   11.999997   11.999973   23.999970   5.99996   0.00962
    6  +   3   -194.535   13.362    0.000    0.000   2.6694   2.6603   2.6649   2.7840    1.000000   11.999996   11.999989   23.999985   6.00000   0.09763
"""
    
    example_levels_2 = """
    0  +   1   -221.955    0.000    0.000    0.000   2.8109   2.7981   2.8045   2.9199    1.000000   12.000000   12.000024   24.000024   0.00001   0.00813
    0  +   2   -209.954   12.001    0.000    0.000   2.8571   2.8459   2.8515   2.9644    1.000000   12.000000   11.999548   23.999548   0.00035   0.02733
    2  +   1   -219.897    0.000    0.000    0.000   2.8140   2.8012   2.8076   2.9229    1.000000   12.000001   12.000014   24.000016   2.00000   0.00357
    2  +   2   -208.304   11.593    0.000    0.000   2.8526   2.8416   2.8471   2.9601    1.000000   12.000005   12.000115   24.000120   2.00000   0.00920
    4  +   1   -215.243    0.000    0.000    0.000   2.8194   2.8067   2.8131   2.9282    1.000000   12.000001   12.000004   24.000005   4.00000   0.00307
    4  +   2   -204.876   10.366    0.000    0.000   2.8436   2.8326   2.8381   2.9515    1.000000   12.000000   11.999958   23.999958   3.99999   0.00695
    6  +   1   -207.898    0.000    0.000    0.000   2.8380   2.8252   2.8316   2.9460    1.000000   12.000001   12.000002   24.000003   6.00000   0.00319
    6  +   2   -199.232    8.666    0.000    0.000   2.8222   2.8106   2.8164   2.9308    1.000000   12.000000   11.999973   23.999973   5.99998   0.00968
    6  +   3   -194.535   13.363    0.000    0.000   2.6694   2.6603   2.6649   2.7841    1.000000   11.999996   11.999989   23.999986   6.00000   0.09765
""" 
    
    example_levels_3 = """
    0  +   1   -239.344    0.000    0.000    0.000   2.6932   2.6824   2.6878   2.8069    1.000000   12.000001   12.000013   24.000014   0.00005   0.01208
    0  +   2   -224.656   14.688    0.000    0.000   2.7845   2.7735   2.7790   2.8945    1.000000   12.000035   12.000425   24.000460   0.00220   0.06782
    2  +   1   -237.315    0.000    0.000    0.000   2.6909   2.6800   2.6854   2.8046    1.000000   12.000008   11.999980   23.999988   2.00001   0.00485
    2  +   2   -223.694   13.621    0.000    0.000   2.7498   2.7389   2.7444   2.8612    1.000000   12.000046   11.999646   23.999692   1.99998   0.07755
    2  +   3   -223.049   14.266    0.000    0.000   2.7420   2.7316   2.7368   2.8537    1.000000   12.000040   12.000100   24.000139   2.00003   0.04992
    4  +   1   -232.808    0.000    0.000    0.000   2.6865   2.6757   2.6811   2.8004    1.000000   12.000000   11.999995   23.999996   3.99999   0.00453
    4  +   2   -220.892   11.916    0.000    0.000   2.6506   2.6410   2.6458   2.7659    1.000000   12.000009   11.999937   23.999945   3.99997   0.05801
    4  +   3   -220.082   12.727    0.000    0.000   2.8233   2.8122   2.8178   2.9319    1.000000   12.000028   11.999996   24.000023   3.99999   0.00578
    6  +   1   -225.545    0.000    0.000    0.000   2.6810   2.6699   2.6755   2.7952    1.000000   11.999993   12.000002   23.999995   6.00001   0.00732
    6  +   2   -217.010    8.535    0.000    0.000   2.6283   2.6200   2.6241   2.7446    1.000000   11.999972   11.999971   23.999943   5.99998   0.02423
    6  +   3   -214.944   10.601    0.000    0.000   2.8156   2.8042   2.8099   2.9245    1.000000   12.000013   11.999987   24.000000   5.99993   0.01089
    6  +   4   -212.160   13.385    0.000    0.000   2.6337   2.6249   2.6293   2.7498    1.000000   11.999985   11.999894   23.999879   5.99995   0.14448
"""
    
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_fewDefs/'
    example_levels_2 = getAllLevelsAsString(
        [f'{MAIN_FLD}K{k}_block_PAV/BU_folder_B1_MZ4_z12n11/HWG/' for k in (1, )])
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_fewDefs/Kmix_block_PAV/BU_folder_B1_MZ4_z12n11/HWG/'
    
    example_levels = getAllLevelsAsString(MAIN_FLD)
     
    # with open("all_spectra_A30_Fermi.txt", 'r') as f:
    #     example_levels_2 = f.read()
    levels_1 = EnergyByJGraph(title='B1')
    levels_1.setData(example_levels, program='taurus_hwg')   
    # #
    levels_2 = EnergyByJGraph(title='HFB sph')
    levels_2.setData(example_levels_2, program='taurus_hwg')  
    
    levels_3 = EnergyByJGraph(title='chiral')
    levels_3.setData(example_levels_3, program='taurus_hwg')  
    
    BaseLevelContainer.RELATIVE_PLOT = True
    BaseLevelContainer.ONLY_PAIRED_STATES = False
    BaseLevelContainer.MAX_NUM_OF_SIGMAS  = 8
    
    _graph = JLevelContainer()
    _graph.global_title = "Comparison HWG D1S from densities"
    _graph.global_title = "23Mg HWG(B1 MZ=4), Mix 2K=1,3,5 "
    _graph.add_LevelGraph(levels_1)
    # _graph.add_LevelGraph(levels_2)
    # _graph.add_LevelGraph(levels_3)
    _graph.plot()
    
    ## ##  EXAMPLE FOR LEVEL GRAPHS ## ##
    levels_1 = EnergyLevelGraph(title='B1')
    levels_1.setData(example_levels, program='taurus_hwg') 
    
    levels_2 = EnergyLevelGraph(title='Fermi')
    levels_2.setData(example_levels_2, program='taurus_hwg') 
    #
    levels_3 = EnergyLevelGraph(title='HFB sph')
    levels_3.setData(example_levels_3, program='taurus_hwg')  
    
    BaseLevelContainer.RELATIVE_PLOT = True
    BaseLevelContainer.ONLY_PAIRED_STATES = False
    BaseLevelContainer.MAX_NUM_OF_SIGMAS  = 6
    
    _graph = BaseLevelContainer()
    _graph.global_title = "23Mg HWG(B1 MZ=4), Mix 2K=1,3,5 "#"Comparison HWG D1S from densities"
    _graph.add_LevelGraph(levels_1)
    # _graph.add_LevelGraph(levels_2)
    # _graph.add_LevelGraph(levels_3)
    _graph.plot()
    
    
    