'''
Created on Dec 23, 2022

@author: Miguel

Module to generate input files for programs.

'''
import numpy as np
from copy import deepcopy, copy

class Enum(object):
    @classmethod
    def members(cls):
        import inspect
        result = []
        for i in inspect.getmembers(cls):
            name = i[0]
            value = i[1]
            if not (name.startswith('_') or inspect.ismethod(value)):
                result.append(value)
        return result

class _Input(object):
    '''
    Abstract class to define inputs for a program
    '''
    _TEMPLATE = """"""
    
    class ArgsEnum:
        pass
    
    def __init__(self, params):
        '''
        Constructor
        '''
        raise InputException("Abstract class, implement me!")
    
    def setParameters(self):
        raise InputException("Abstract class, implement me!")
    
    def setConstraints(self):
        raise InputException("Abstract class, implement me!")
    
    def __str__(self):
        return self._TEMPLATE.format(**self.getdefault())
    
    def copy(self):
        return deepcopy(self)

class InputException(BaseException):
    pass

class TaurusInput(_Input):
    
    '''
        Class to set the template for the program Taurus
    '''
    
    _TEMPLATE = """Interaction
-----------
Master name hamil. files      {interaction}
Center-of-mass correction     {com}
Read reduced hamiltonian      {red_hamil}
No. of MPI proc per H team    0

Particle Number
---------------
Number of active protons      {z}.00
Number of active neutrons     {n}.00
No. of gauge angles protons   {z_Mphi}
No. of gauge angles neutrons  {n_Mphi}

Wave Function
-------------
Type of seed wave function    {seed}
Number of QP to block         {qp_block}
No symmetry simplifications   0
Seed random number generation 0
Read/write wf file as text    0
Cutoff occupied s.-p. states  0.00E-00
Include all empty sp states   0
Spatial one-body density      {pnt_dens}
Discretization for x/r        {discr_xr}
Discretization for y/theta    {discr_yt}
Discretization for z/phi      {discr_zp}

Iterative Procedure
-------------------
Maximum no. of iterations     {iterations}
Step intermediate wf writing  {interm_wf}
More intermediate printing    0
Type of gradient              {grad_type}
Parameter eta for gradient    {eta_grad:4.3E}
Parameter mu  for gradient    {mu_grad:4.3E}
Tolerance for gradient        {grad_tol:4.3E}

Constraints
-----------
Force constraint N/Z          1
Constraint beta_lm            1
Pair coupling scheme          {pair_schm}
Tolerance for constraints     1.000E-08
Constraint multipole Q10      {b10}
Constraint multipole Q11      {b11}
Constraint multipole Q20      {b20}
Constraint multipole Q21      {b21}
Constraint multipole Q22      {b22}
Constraint multipole Q30      {b30}
Constraint multipole Q31      {b31}
Constraint multipole Q32      {b32}
Constraint multipole Q33      {b33}
Constraint multipole Q40      {b40}
Constraint multipole Q41      {b41}
Constraint multipole Q42      {b42}
Constraint multipole Q43      {b43}
Constraint multipole Q44      {b44}
Constraint radius sqrt(r^2)   {sqrt_r2}
Constraint ang. mom. Jx       {Jx}
Constraint ang. mom. Jy       {Jy}
Constraint ang. mom. Jz       {Jz}
Constraint pair P_T00_J10     {P_T00_J10}
Constraint pair P_T00_J1m1    {P_T00_J1m1}
Constraint pair P_T00_J1p1    {P_T00_J1p1}
Constraint pair P_T10_J00     {P_T10_J00}
Constraint pair P_T1m1_J00    {P_T1m1_J00}
Constraint pair P_T1p1_J00    {P_T1p1_J00}
Constraint field Delta        0   0.000"""
    
    class ArgsEnum(Enum):
        interaction    = 'interaction'
        com            = 'com'
        red_hamil = 'red_hamil' 
        z        = 'z'
        n        = 'n'
        z_Mphi   = 'z_Mphi'
        n_Mphi   = 'n_Mphi'
        seed     = 'seed'
        qp_block = 'qp_block'
        # qparticle_blocking = 'qparticle_blocking'
        pnt_dens = 'pnt_dens'
        discr_xr = 'discr_xr'
        discr_yt = 'discr_yt'
        discr_zp = 'discr_zp'
        
        iterations = 'iterations'
        interm_wf  = 'interm_wf'
        grad_type  = 'grad_type'
        eta_grad   = 'eta_grad'
        mu_grad    = 'mu_grad'
        grad_tol   = 'grad_tol'
        pair_schm  = 'pair_schm'
    
    class ConstrEnum(Enum):
        
        b10 = 'b10'
        b11 = 'b11'
        b20 = 'b20'
        b21 = 'b21'
        b22 = 'b22'
        b30 = 'b30'
        b31 = 'b31'
        b32 = 'b32'
        b33 = 'b33'
        b40 = 'b40'
        b41 = 'b41'
        b42 = 'b42'
        b43 = 'b43'
        b44 = 'b44'
        sqrt_r2 = 'sqrt_r2'
        Jx = 'Jx'
        Jy = 'Jy'
        Jz = 'Jz'
        P_T00_J10  = 'P_T00_J10'
        P_T00_J1m1 = 'P_T00_J1m1'
        P_T00_J1p1 = 'P_T00_J1p1'
        P_T10_J00  = 'P_T10_J00'
        P_T1m1_J00 = 'P_T1m1_J00'
        P_T1p1_J00 = 'P_T1p1_J00'
    
    _TEMPLATE_INPUT_DD = """* Density dep. Interaction:    ------------
eval_density_dependent (1,0)= {eval_dd}
eval_rearrangement (1,0)    = {eval_rea}
eval_explicit_fieldsDD (1,0)= {eval_explicit_dd}
t3_DD_CONST [real  MeV]     = {t3_param:8.6e}
x0_DD_FACTOR                = {x0_param:8.6e}
alpha_DD                    = {alpha_param:8.6f}
* Integration parameters:      ------------
*  0 trapezoidal, 1 Gauss-Legendre, 2 Gauss-Laguerre(r)/Legendre, 3 Laguerre-Lebedev
integration_method (0,1,2,3)= 3
export_density (1, 0)       = {eval_export_h}
r_dim                       = {r_dim}
Omega_Order                 = {omega_dim}
THE_grid                    = 10
PHI_grid                    = 10
R_MAX                       = 08.500000d+00
eval full Val.Space (0,1)   = 1
* Integration parameters:      ------------"""
    
    class InpDDEnum(Enum):
        eval_dd = 'eval_dd'
        eval_rea = 'eval_rea'
        eval_explicit_dd = 'eval_explicit_dd'
        eval_export_h  = 'eval_export_h'
        x0_param = 'x0_param'
        t3_param = 't3_param'
        alpha_param = 'alpha_param'
        r_dim = 'r_dim'
        omega_dim = 'omega_dim'
    
    _DEFAULT_DD_PARAMS = {
        InpDDEnum.eval_dd     : 1,
        InpDDEnum.eval_rea    : 1, 
        InpDDEnum.eval_explicit_dd : 0,
        InpDDEnum.eval_export_h    : 0, 
        InpDDEnum.x0_param    : 1.0,
        InpDDEnum.t3_param    : 1390.6,
        InpDDEnum.alpha_param : 0.333333,
        InpDDEnum.r_dim       : 10,
        InpDDEnum.omega_dim   : 10,
        }
    
    def __init__(self, z, n, interaction, **params):
        """
        Construct a input object for taurus, several parameters can be given.
        Arguments z, n, interaction are mandatory 
        """
        self.z = z
        self.n = n
        self.interaction = interaction
        
        ## default values
        self.com        = 0
        self.red_hamil  = 0
        self.z_Mphi     = 1
        self.n_Mphi     = 1
        self.seed       = 0
        self.qp_block   = 0
        self.pnt_dens   = 0
        self.discr_xr   = (0, 0.0)
        self.discr_yt   = (0, 0.0)
        self.discr_zp   = (0, 0.0)
        self.iterations = 0
        self.interm_wf  = 1
        self.grad_type  = 1
        self.eta_grad   = 0.001
        self.mu_grad    = 0.2
        self.grad_tol   = 0.01
        self.pair_schm  = 1
        
        ## default constraints (b1 constraints = 1 lead to w.f to slide by the 
        ## COM, setted by default to 0, in case of core-calc, remember to reset them
        # DONE: define methods to select default configurations: 
        # i.e, core-calculation: Q1*, 3* = (0, 0), 
        ## NOTE: when constraints at None will print '0   0.000' 
        self.b10        = 0.0
        self.b11        = 0.0
        self.b20        = None
        self.b21        = 0.0
        self.b22        = None
        self.b30        = None
        self.b31        = 0.0
        self.b32        = None
        self.b33        = None
        self.b40        = None
        self.b41        = 0.0
        self.b42        = None
        self.b43        = None
        self.b44        = None
        self.sqrt_r2    = None
        self.Jx         = None
        self.Jy         = None
        self.Jz         = None
        self.P_T00_J10  = None
        self.P_T00_J1m1 = None
        self.P_T00_J1p1 = None
        self.P_T10_J00  = None
        self.P_T1m1_J00 = None
        self.P_T1p1_J00 = None
        
        self.setParameters(**params)
    
    def setParameters(self, **params):
        """
        both internal and interface method to modify several constraints 
        Use recommended to apply assertions over the valid taurus parameters
        """
        constraints = {}
        for arg, value in params.items():
            if   hasattr(self.ConstrEnum, arg):
                constraints[arg] = value 
                continue
            elif hasattr(self.ArgsEnum, arg):
                ## Case selection of the non-constrained parts INLINE TEST
                if   arg in (self.ArgsEnum.com,       self.ArgsEnum.red_hamil,
                             self.ArgsEnum.interm_wf, self.ArgsEnum.pair_schm):
                    assert value in (0,1), f"Value must be 1 or 0 [{value}]"
                elif arg in (self.ArgsEnum.z_Mphi,   self.ArgsEnum.n_Mphi,
                             self.ArgsEnum.qp_block, self.ArgsEnum.iterations):
                    assert type(value)==int and value >=0, \
                        f"Value must be non-negative integer [{value}]"
                elif arg == self.ArgsEnum.seed:
                    assert type(value)==int and value >=0 and value < 10, \
                        f"Value must be in range 0-9 [{value}]"
                elif arg == self.ArgsEnum.pnt_dens:
                    assert type(value)==int and value >=0 and value < 4, \
                        f"Value must be in range 0-3 [{value}]"
                elif arg == self.ArgsEnum.grad_type:
                    assert type(value)==int and value >=0 and value < 3, \
                        f"Value must be in range 0-2 [{value}]"
                elif arg in (self.ArgsEnum.eta_grad,   self.ArgsEnum.mu_grad,
                             self.ArgsEnum.grad_tol):
                    assert type(value)==float and value >=0.0, \
                        f"Value must be non-negative float [{value}]"
                elif arg.startswith(self.ArgsEnum.discr_xr[:6]):
                    assert isinstance(value, tuple), \
                        "Argument given to the discretization for density must be tuple"
                    assert type(value[0])==int and value[0] >=0, \
                        f"Value must be non-negative integer [{value}]"
                    assert type(value[1])==float and value[1] >=0.0, \
                        f"Value must be non-negative float [{value}]"
                else:
                    assert type(value)==str, "Interaction hamil must be string"
                    
                setattr(self, arg, value)
                    
            else:
                raise(f"Unidentified argument to set: [{arg}] val[{value}]")
        
        if len(constraints) > 0:
            self.setConstraints(**constraints)
    
    
    def _discretizationArgs2Text(self):
        """ 
        Returns text of the parameters for the spatial density set: Nx dx, ...
        """
        N_xr, dxr = self.discr_xr
        N_yt, dyt = self.discr_yt
        N_zp, dzp = self.discr_zp
        return [(self.ArgsEnum.discr_xr, f"{N_xr:<3} {dxr:4.3f}"),
                (self.ArgsEnum.discr_yt, f"{N_yt:<3} {dyt:4.3f}"),
                (self.ArgsEnum.discr_zp, f"{N_zp:<3} {dzp:4.3f}")]
        
    def setDisctretizationArgs(self, dens_mode, N_xr, N_yt, N_zp, dxr, dyt, dzp):
        """
        set the attributes for the density discretization, calling the function
        require the setting of one type of spatial_density mode:
            0: No spatial_dens 
            1: Only radial,
            2: Radial and Angular density:, dyt(zp) = pi / N_('') on taurus
            3: XYZ, cartesian
        """
        self.pnt_dens = dens_mode
        self.discr_xr = (0, 0.0)
        self.discr_yt = (0, 0.0)
        self.discr_zp = (0, 0.0)
        if   (dens_mode == 1):
            assert type(N_xr)== int and type(dxr)==float, InputException(N_xr,dxr)
            self.discr_xr = (N_xr, dxr)
        elif (dens_mode == 2):
            assert type(N_xr)== int and type(dxr)==float, InputException(N_xr,dxr)
            assert type(N_yt)== int, InputException(N_yt, dyt)
            assert type(N_zp)== int, InputException(N_zp, dzp)
            self.discr_xr = (N_xr, dxr)
            self.discr_yt = (N_yt,   np.pi / N_yt) ## default dth for taurus
            self.discr_zp = (N_zp, 2*np.pi / N_zp) ## default
        elif (dens_mode == 3):
            assert type(N_xr)== int and type(dxr)==float, InputException(N_xr,dxr)
            assert type(N_yt)== int and type(dyt)==float, InputException(N_yt,dyt)
            assert type(N_zp)== int and type(dzp)==float, InputException(N_zp,dzp)
            self.discr_xr = (N_xr, dxr)
            self.discr_yt = (N_yt, dyt)
            self.discr_zp = (N_zp, dzp)
        else:
            raise InputException(f" dens_mode must be = 0,1,2,3. Got[{dens_mode}]")
            
    
    def _quasipart2block(self):
        """
        Set the Qp to block, attr self.qp_block is the qp state for the blocking,
        """
        if self.qp_block not in (None, 0):
            if isinstance(self.qp_block, str):
                assert self.qp_block.isdigit(), InputException(
                    " ".join(["Got", self.qp_block, ", is not integer"]))
            assert isinstance(self.qp_block, int), InputException(
                " ".join(["Must give an integer or int-string, got:", self.qp_block]))
            return f"1\nQP_blocked_1                  {self.qp_block}"
        else:
            return '0'
    
    @staticmethod
    def getIndexForQuasiPartFromShellList(sh_state, m_j, m_t, hamil_shell_list, 
                                          l_ge_10=True):
        """ 
        Given a H.O shell states from a .sho file (the order followed by taurus
        for the uncoupled base internally), get the index of that state to be 
        blocked. Exception will be risen if the value is not valid."""
        index_ = 0
        total_deg = 0
        state_found = False
        
        for spss in hamil_shell_list:
            n, l, j = int(spss)
            deg = j + 1
            if not state_found:
                if l_ge_10:
                    st_ge_10 = str((10000*n) + (1000*l) + j)
                    state_found = int(sh_state) == st_ge_10
                else:
                    st_le_10 = str((1000*n) + (1000*l) + j) ## deprecated
                    state_found = int(sh_state) == st_le_10
                
                if state_found:
                    _j_sts = [i for i in range(-j, j+1, 2)]
                    if m_j in _j_sts:
                        index_ = total_deg + _j_sts.index(m_j) + 1
                    else:
                        print(f"Warning!, m_j state[{sh_state}, {m_j}] not in the j={j}/2 shell. Setting m=-j" )
                        index_ = total_deg + 1
            total_deg += deg
        
        if not state_found:
            raise InputException(f"The state [{sh_state},{m_j}] could not be found in [{hamil_shell_list}], check it out!")
        
        if m_t == 1: ## neutron states append the proton data dimension
            index_ += total_deg
        return index_
    
    def _constraint2text(self, cons_key):
        """
        Set text for deformations defined
        """
        assert hasattr(self.ConstrEnum, cons_key), f"Invalid constraint. Got [{cons_key}]"
            
        if (getattr(self, cons_key) != None):
            val = getattr(self, cons_key)
            if  isinstance(val, tuple) and len(val)==2:
                assert cons_key.startswith('b') or (cons_key==self.ConstrEnum.sqrt_r2),\
                    InputException( "Only r^2 or multipoles are allowed to "
                                   f"separate p/n constraints. [{cons_key}]")
                return f'2  {val[0]:+6.3f}  {val[1]:+6.3f}'
            elif isinstance(val, float):
                return f'1  {val:+6.3f}'
            else:
                raise InputException(f"WTF did you give in [{cons_key}]? [{val}]")
        # case: the value is not constrained
        return '0   0.000'
    
    @classmethod
    def set_inputDDparamsFile(cls, **params):
        ''' 
        !! Classmethod, modifications using this will remain for every instance
        of TaurusInput
        Args:
            :eval_dd, rea, explicit_dd : 0,1
            :x0_param, t3_param, alpha_param : <float>
            :r_dim, omega_dim  :: <int>
        '''
        for arg, val in params.items():
            if arg not in cls.InpDDEnum.members():
                print(f"[WARNING] parameter DD unidentified: [{arg}, {val}]")
                continue
            ## valid argument, check invalid value
            if arg.startswith('eval_'):
                assert val in (0,1), \
                    f"attrib eval_dd/rea/explicit_dd must be 0,1. Got[{val}]"
            elif arg.endswith('_param'):
                assert isinstance(val, (float, int)), \
                    f"DD params must be float(or int). Got[{val}]"
            if arg.endswith('_dim'):
                assert type(val) == int, \
                    f"R/Omega dim must be positive integers. Got[{val}]"
            
            cls._DEFAULT_DD_PARAMS[arg] = val
            
    def get_inputDDparamsFile(self, r_dim=None, omega_dim=None):
        ''' '''
        params = copy(self._DEFAULT_DD_PARAMS)
        
        if r_dim and isinstance(r_dim, int):
            params[self.InpDDEnum.r_dim] = r_dim
        if omega_dim and isinstance(omega_dim, int):
            params[self.InpDDEnum.omega_dim] = omega_dim
        
        if ((params[self.InpDDEnum.eval_export_h] == 1) or 
            (params[self.InpDDEnum.eval_explicit_dd] == 1)):
            print("[WARNING] eval_calculations will use explicit evaluation of "
                  "the matrix elements and the time required will grow exponentially.")
        
        txt_ = self._TEMPLATE_INPUT_DD.format(**params)
        txt_ = txt_.replace('e+', 'd+')
        txt_ = txt_.replace('e-', 'd-')
        return txt_
        
    def setConstraints(self, **constraints):
        """
        Set constraints, remember that only multipoles and radius can be set a 
        separate constraint for protons and neutrons (given by a 2-tuple), 
        Common constraint must be a float to continue, set None to invalidate 
        previous modifications.
        """
        for constr, value in constraints.items(): 
            assert hasattr(self.ArgsEnum, constr), f"Invalid constraint. Got [{constr}]"
            
            if isinstance(value, tuple) and len(value)==2:
                assert constr.startswith('b') or (constr==self.ArgsEnum.sqrt_r2),\
                    InputException( "Only r^2 or multipoles are allowed to "
                                   f"separate p/n constraints. [{constr}]")
                setattr(self, constr, (float(value[0]), float(value[1])) )
            elif isinstance(value, float) or value==None:
                setattr(self, constr, value)
            else:
                raise InputException( "Invalid constraint format, got:"
                                     f"c[{constr}], val[{value}]")
    
    def getText4file(self):
        """
        Return of the Input template filled
        """
        ## get attribute values
        kwargs = []
        for k_ in self.ArgsEnum.members():
            if k_.startswith(self.ArgsEnum.discr_xr[:5]):
                kwargs += self._discretizationArgs2Text()
            elif  k_ == self.ArgsEnum.qp_block:
                kwargs.append((self.ArgsEnum.qp_block, self._quasipart2block()))
            else:
                kwargs.append( (k_, getattr(self, k_)) )
        
        for k_ in self.ConstrEnum.members():
            kwargs.append( (k_, self._constraint2text(k_)) )
        
        kwargs = dict(kwargs)
        txt_ = self._TEMPLATE.format(**kwargs)
        return txt_
    
    def __str__(self):
        return self.getText4file()
    
    def setUpValenceSpaceCalculation(self, **params):
        """ 
        Set up for the Valence Space configuration. Odd Multipoles must be zero
        """
        self.com = 0
        self.b10 = 0.0
        self.b11 = 0.0
        self.b21 = None
        self.b30 = 0.0
        self.b31 = 0.0
        self.b32 = 0.0
        self.b33 = 0.0
        self.b41 = None
        
        self.setParameters(**params)
    
    def setUpNoCoreAxialCalculation(self, **params):
        """ 
        Set up for the No-Core configuration. PN pairing must be suppressed, and
        default seed is 3, requires COM correction
        """
        
        self.com  = 1
        self.seed = 3
        for atr in (self.ConstrEnum.b10, self.ConstrEnum.b11,
                    self.ConstrEnum.b21, self.ConstrEnum.b22,self.ConstrEnum.b44,
                    self.ConstrEnum.b31, self.ConstrEnum.b32, self.ConstrEnum.b33,
                    self.ConstrEnum.b41, self.ConstrEnum.b42, self.ConstrEnum.b43,
                    self.ConstrEnum.P_T00_J10,  self.ConstrEnum.P_T00_J1m1,
                    self.ConstrEnum.P_T00_J1p1, self.ConstrEnum.P_T10_J00):
            setattr(self, atr, 0.0)
        
        self.setParameters(**params)
    
    def copy(self):
        """ return a copy of a previous element """
        snd_inp = TaurusInput(self.z, self.n, self.interaction)
        for atr in (*self.ArgsEnum.members(), *self.ConstrEnum.members()):
            setattr(snd_inp, atr, getattr(self, atr))
        return snd_inp



class AxialInput(_Input):
    """
    Class for HFBAxial code
    """
    
    _TEMPLATE = """NUCLEUS    {a:03} XX   Z= {z:03}     >>> HFB OPT <<< {com} COULECH {coul}
EPSG {grad_tol:8.6f} MAXITER {iterations:05}    >>> OUTPUT  <<< 0 **** TSTG 0
ETMAX 0.7501 ETMIN 0.0351 DMAX 0.90 DMIN 0.70 TSHH 399.0
GOGNY FORCE        {interaction}    *** 0=D1S 1=D1 2=D1' 3(t3=0)
INPUT W.F.         {seed}    *** 0,1=WF FROM UNIT 10 (1 kicks), 2=NEW Function
OSCILLATOR LENGHT  0    *** 0               BP {b_len:9.7f} BZ {b_len:9.7f}
          >>>>>>>>>> C O N S T R A I N T S <<<<<<<<<<<
{_constraints}          >>>>>>>>>> E N D <<<<<<<<<<<<<<<<<<<<<<<<<<<  """
    
    class ArgsEnum(Enum):
        interaction    = 'interaction'
        com        = 'com'
        coul       = 'coul'
        z          = 'z'
        a          = 'a'
        seed       = 'seed'
        iterations = 'iterations'
        grad_tol   = 'grad_tol'
        b_len      = 'b_len'
        _constraints = '_constraints'
    
    class ConstrEnum(Enum):
        b10 = 'b10'
        b20 = 'b20'
        b30 = 'b30'
        b40 = 'b40'
        q10 = 'q10'
        q20 = 'q20'
        q30 = 'q30'
        q40 = 'q40'
        sqrt_r2 = 'sqrt_r2'
        var_jx = 'var_jx'
        var_n  = 'var_n'
    
    _COM_ARGS       = "CM1 {} CM2 {}"
    _COM_CONSRT     = "C.O.M.     1 1   0.00000000D+00"
    _QL_CONSRT_TEMPL   = "QL    {:1}    {:1} {:1}   {:10.8f}D+00"
    _BL_CONSTR_TEMPL   = "BL    {:1}    {:1} {:1}   {:10.8f}D+00"
    _DN2_CONSTR_TEMPL  = "DN**2      {:1} {:1}   {:10.8f}D+00"
    _DJX2_CONSTR_TEMPL = "DJX**2     {:1} {:1}   {:10.8f}D+00"
    _R2_CONSTR_TEMPL   = "<R**2>     {:1} {:1}   {:10.8f}D+00"
    
    
    def __init__(self, z, n, interaction, **params):
        
        self.z = z
        self.a = n + z
        self.interaction = interaction
        
        ## default values
        self.com        = 0
        self.coul       = 2 # 0 no, 1 Slater determinant, 2 exact Coul Exchange
        self.seed       = 0
        self.iterations = 500
        self.grad_tol   = 0.000001
        self.b_len      = 1.0
        self.b10        = None
        self.b20        = None
        self.b30        = None
        self.b40        = None
        self.q10        = None
        self.q20        = None
        self.q30        = None
        self.q40        = None
        self.sqrt_r2    = None
        self.var_jx     = None
        self.var_n      = None
        
        self.setParameters(**params)
    
    def setParameters(self, **params):
        """
        both internal and interface method to modify several constraints 
        Use recommended to apply assertions over the valid taurus parameters
        """
        constraints = {}
        for arg, value in params.items():
            if   hasattr(self.ConstrEnum, arg):
                constraints[arg] = value 
                continue
            elif hasattr(self.ArgsEnum, arg):
                ## Case selection of the non-constrained parts INLINE TEST
                if   arg in (self.ArgsEnum.com, self.ArgsEnum.coul,
                             self.ArgsEnum.seed):
                    assert value in (0,1,2), f"Value must be in range 0-2 [{value}]"
                elif arg in (self.ArgsEnum.iterations, 
                             self.ArgsEnum.z, self.ArgsEnum.a):
                    assert type(value)==int and value >=0, \
                        f"Value must be non-negative integer [{value}]"
                    if arg == self.ArgsEnum.iterations:
                        value = min(value, 99999) ## to not extend the line in template
                elif arg == self.ArgsEnum.interaction:
                    assert type(value)==int and value >=0 and value < 10, \
                        f"Value must be in range 0-9 [{value}]"
                elif arg == self.ArgsEnum.grad_tol:
                    assert type(value)==float and value > 9.9e-7, \
                        f"Value must be non-negative float greater than 1e-6 [{value}]"
                elif arg == self.ArgsEnum.b_len:
                    assert type(value)==float and value >= 0.0, \
                        f"Value must be non-negative float [{value}]"
                else:
                    raise InputException(f"Unidentified argument to set: [{arg}] val[{value}]")
                    
                setattr(self, arg, value)
                    
            else:
                raise InputException(f"Unidentified argument to set: [{arg}] val[{value}]")
        
        if len(constraints) > 0:
            self.setConstraints(**constraints)
    
    def _getCom1B2BArgSwitch(self):
        if   self.com == 0:
            return self._COM_ARGS.format(0,0)
        elif self.com == 1:
            return self._COM_ARGS.format(1,0)
        elif self.com == 2:
            return self._COM_ARGS.format(1,1)
        else:
            raise InputException(f"COM body selector is not 0,1,2. got [{self.com}]")
    
    def _setConstrTemplates(self, constr, pp, nn, val):
        if   constr.startswith('b'):
            L = int(constr[1])
            return self._BL_CONSTR_TEMPL.format(L, pp, nn, val)
        elif constr.startswith('q'):
            L = int(constr[1])
            return self._QL_CONSTR_TEMPL.format(L, pp, nn, val)
        elif constr == self.ConstrEnum.var_n:
            return self._DN2_CONSTR_TEMPL.format(pp, nn, val)
        elif constr == self.ConstrEnum.var_jx:
            return self._DJX2_CONSTR_TEMPL.format(pp, nn, val)
        elif constr == self.ConstrEnum.sqrt_r2:
            return self._R2_CONSTR_TEMPL.format(pp, nn, val)
        else:
            raise InputException(f"Unidentified constraint given [{constr}]")
        
    
    def _constraint2text(self, cons_key):
        """
        Set text for deformations defined, return the template filled
        """
        assert hasattr(self.ConstrEnum, cons_key), f"Invalid constraint. Got [{cons_key}]"
            
        if (getattr(self, cons_key) != None):
            val = getattr(self, cons_key)
            if  isinstance(val, tuple) and len(val)==2:
                ## case proton and neutrons separately
                _1 = self._setConstrTemplates(cons_key, 1, 0, val[0])
                _2 = self._setConstrTemplates(cons_key, 0, 1, val[1])
                return _1+"\n"+_2
                
            elif isinstance(val, float):
                return self._setConstrTemplates(cons_key, 1, 1, val)
            else:
                raise InputException(f"WTF did you give in [{cons_key}]? [{val}]")
        # case: the value is not constrained
        return ''
    
    def setConstraints(self, **constraints):
        """
        Set constraints, remember that only multipoles and radius can be set a 
        separate constraint for protons and neutrons (given by a 2-tuple), 
        Common constraint must be a float to continue, set None to invalidate 
        previous modifications.
        """
        for constr, value in constraints.items(): 
            assert hasattr(self.ConstrEnum, constr), f"Invalid constraint. Got [{constr}]"
            
            if isinstance(value, tuple) and len(value)==2:
                setattr(self, constr, (float(value[0]), float(value[1])) )
            elif isinstance(value, float) or value==None:
                setattr(self, constr, value)
            else:
                raise InputException( "Invalid constraint format, got:"
                                     f"c[{constr}], val[{value}]")
    
    def getText4file(self):
        """
        Return of the Input template filled
        """
        ## get attribute values
        kwargs = []
        for k_ in self.ArgsEnum.members():
            if k_ == self.ArgsEnum.com: ## set 1b or 2b com has a template
                kwargs.append( (k_, self._getCom1B2BArgSwitch()) )
                continue
            kwargs.append( (k_, getattr(self, k_)) )
            
        constr_block = [self._COM_CONSRT, ] # com fixing must always be
        for k_ in self.ConstrEnum.members():
            if getattr(self, k_) == None: continue
            constr_block.append( self._constraint2text(k_) )
        
        kwargs = dict(kwargs)
        kwargs[self.ArgsEnum._constraints] = '\n'.join(constr_block) + '\n'
        txt_ = self._TEMPLATE.format(**kwargs)
        return txt_
    
    def __str__(self):
        return self.getText4file()
    

#===============================================================================
# TESTS
#===============================================================================
# t_input = TaurusInput(10, 12, 'hamil')
# t_input.setParameters(**{TaurusInput.ArgsEnum.seed : 3,
#                        TaurusInput.ArgsEnum.iterations : 912, 
#                        TaurusInput.ArgsEnum.com  : 1,
#                        TaurusInput.ArgsEnum.discr_zp : (33, 0.01)})
# # t_input.setDisctretizationArgs(1, 100, 0, 0, 0.05, 0, 0)
# t_input.b20 = (1.2, 1.23)
# print(t_input.getText4file())

# a_inp = AxialInput(10, 12, 3, b_len=1.2)
# a_inp.setParameters(**{AxialInput.ArgsEnum.com : 1,
#                         AxialInput.ArgsEnum.seed: 2,
#                         AxialInput.ArgsEnum.iterations: 15163,
#                         AxialInput.ConstrEnum.b20 : 1.02,
#                         AxialInput.ConstrEnum.sqrt_r2: (4.3212345678, 
#                                                         3.1234567890)
#                         })
# a_inp.var_n = (2.365, 0.000)
# print(a_inp)
#
#
t_input = TaurusInput(2,4, 'hamil')
TaurusInput.set_inputDDparamsFile(**{
    TaurusInput.InpDDEnum.eval_explicit_dd : 1,
    TaurusInput.InpDDEnum.t3_param         : 100.0,
    TaurusInput.InpDDEnum.x0_param         : 0.0000000001})
print(t_input.get_inputDDparamsFile(11, 20))
