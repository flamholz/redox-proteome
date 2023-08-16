import pandas as pd
import numpy as np
import cvxpy as cp

from collections import defaultdict

"""
Class uses resource balance/FBA style approach to optimize resource 
allocation in a simple 3 half reaction model of a single metabolism. 
"""

__author__ = "Avi I. Flamholz"

# Constants
# Cell mass density assumed constant.
RHO_CELL = 1000.0 # g/L
# molar mass of a carbon atom  
MW_C_ATOM = 12.0    
S_PER_HR = 60*60 

# 70%  of cell mass is water, 30% other stuff. 
WATER_FRACTION = 0.7
DW_FRACTION = 1-WATER_FRACTION
# gDW per gC -- dry mass is 50% carbo. 
GDW_PER_G_C = 2                   
# g cells per g carbon
GCELL_PER_GC = GDW_PER_G_C/DW_FRACTION  


class RateLawFunctor(object):
    """Abstract base class for rate law functors."""

    ORDER = None

    def Apply(self, S, processes, metabolites, gammas, phis, concs):
        """Applies the rate law to the given concentrations.

        Args:
            S: numpy array, stoichiometric matrix.
                Reactions on rows, metabolites on cols. 
            processes: pd.Series, process names in order of S.
            metabolites: pd.Series, metabolite names in order of S.
            gammas: numpy array, gamma values.
            phis: numpy array, phi values.
            concs: numpy array, concentrations.

        Returns:
            numpy array, reaction fluxes.
        """
        raise NotImplementedError
    

class ZerothOrderRateLaw(RateLawFunctor):
    """Zeroth order rate law. 

    Flux is independent of the concentrations of the reactants.
    """
    ORDER = 0

    def Apply(self, S, processes, metabolites, gammas, phis, concs):
        return cp.multiply(gammas, phis)


class SimpleFirstOrderRateLaw(RateLawFunctor):
    """First order rate law. 

    Fluxes depends on concentrations to the first power.

    Here anabolism is first order in ATP and reduction is first order in NADH.
    """
    ORDER = 1

    def Apply(self, S, processes, metabolites, gammas, phis, concs):
        m_list = list(metabolites)
        p_list = list(processes)

        ATP_index = m_list.index('ATP')
        NADH_index = m_list.index('ECH')  # generic 2 e- carrier, reduced
        NAD_index = m_list.index('EC')    # generic 2 e- carrier, oxidized
        ana_index = p_list.index('anabolism')
        ox_index = p_list.index('oxidation')
        red_index = p_list.index('reduction')
        h_index = p_list.index('ATP_homeostasis')

        # binary matrix -- processes x metabolites -- indicating where ATP and NADH 
        # are substrates. Note, for now each reaction can only have one substrate.
        subs = np.zeros(S.shape)
        subs[ana_index, ATP_index] = 1
        subs[red_index, NADH_index] = 1
        subs[ox_index, NAD_index] = 1
        subs[h_index, ATP_index] = 1

        conc_term = subs @ concs
        return cp.multiply(gammas, cp.multiply(phis, conc_term))
    

class GrowthRateOptParams(object):
    """A class to hold the parameters for the growth rate optimization problem.

    Attributes:
        do_dilution: boolean, whether to include dilution in the model.
        do_maintenance: boolean, whether to include maintenance in the model.
        rate_law: RateLawFunctor, rate law to use.
        min_phi_O: float, minimum mass fraction for other processes.
        phi_O: float, mass fraction for other processes.
            Only one of min_phi_O and phi_O should be set.
        maintenance_cost: float, maintenance cost. Units of [mmol ATP/gDW/hr].
                These are typically reported units for convenience.
        ATP_maint: float, maintenance in units of [molar ATP/s].
        max_lambda_hr: float, maximum lambda value. Units of [1/hr].
        fixed_ATP: float, fixed ATP concentration. [mol/L] units.
        fixed_NADH: float, fixed NADH concentration. [mol/L] units.
    """
    def __init__(self, do_dilution=False, rate_law=None,
                 min_phi_O=None, phi_O=None,
                 maintenance_cost=0, max_lambda_hr=None,
                 fixed_ATP=None, fixed_NADH=None,
                 fixed_ra=None, fixed_re=None):
        """Initializes the GrowthRateOptParams class.

        Args:
            do_dilution: boolean, whether to include dilution in the model.
            rate_law: RateLawFunctor, rate law to use. If None, uses zeroth order.
            min_phi_O: float, minimum mass fraction for other processes.
            phi_O: float, mass fraction for other processes.
                Only one of min_phi_O and phi_O should be set.
            maintenance_cost: float, maintenance cost. Units of [mmol ATP/gDW/hr].
                These are typically reported units for convenience.
            max_lambda_hr: float, maximum lambda value. Units of [1/hr].
            fixed_ATP: float, fixed ATP concentration. [mol/L] units.
            fixed_NADH: float, fixed NADH concentration. [mol/L] units.
            fixed_re: float, fixed ratio of NAD/NADH concentrations.
            fixed_ra: float, fixed ratio of ADP/ATP concentrations.
        """
        msg = "Only one of min_phi_O and phi_O should be set."
        assert (min_phi_O is None) or (phi_O is None), msg

        if do_dilution:
            msg = "Must define concentrations for dilution"
            assert (fixed_ATP is not None) and (fixed_NADH is not None), msg

        self.rate_law = rate_law or ZerothOrderRateLaw()
        if self.rate_law.ORDER > 0:
            msg = "Concentration dependent rate laws require ATP and NADH concentrations."
            assert (fixed_ATP is not None) and (fixed_NADH is not None), msg

        self.do_dilution = do_dilution
        self.min_phi_O = min_phi_O or 0
        self.phi_O = phi_O
        self.max_lambda_hr = max_lambda_hr
        self.fixed_ATP = fixed_ATP or 0
        self.fixed_ra = fixed_ra or 0
        self.fixed_ADP = self.fixed_ATP * self.fixed_ra
        self.fixed_NADH = fixed_NADH or 0
        self.fixed_re = fixed_re or 0
        self.fixed_NAD = self.fixed_NADH * self.fixed_re

        # Convert maintenance molar ATP/s
        self.maintenance_cost = maintenance_cost or 0
        m = 1e-3*(self.maintenance_cost)/S_PER_HR   # mol ATP/gDW/s
        # mol ATP/gDW/s * gDW/g cell * g cell/L = molar ATP/s
        self.ATP_maint = m * DW_FRACTION * RHO_CELL
        

class LinearMetabolicModel(object):
    """Coarse-grained linear model of resource allocation while balancing ATP and e- fluxes."""

    @classmethod
    def FromFiles(cls, metabolites_fname, stoich_fname):
        m_df = pd.read_csv(metabolites_fname, index_col=0)
        
        S_df = pd.read_csv(stoich_fname, index_col=0)
        
        # Pandas prefers integers by default. If we have no float stoichiometries,
        # we have to convert to floats ... we will need floats for fractional
        # coefficients that arise in changing the NOSC
        dtype_dict = {}
        for k,dt in S_df.dtypes.items():
            if dt in (np.int8, np.int16, np.int32, np.int64): 
                dt = np.float64
            dtype_dict[k] = dt
            
        # Return an instance
        return cls(m_df, S_df.astype(dtype_dict))
    
    def __init__(self, metabolites_df, S_df):
        self.m_df = metabolites_df
        self.S_df = S_df
        
        # Record the ZC values
        self.ZCorg = self.m_df.loc['C_red'].NOSC
        self.ZCprod = self.m_df.loc['C_ox'].NOSC
        self.ZCB = self.m_df.loc['biomass'].NOSC
        
        # make a numeric stoichiometric matrix
        self._update_S()
        # check C and e- balancing
        self._check_c_balance()
        self._check_e_balance()

    def copy(self):
        return LinearMetabolicModel(self.m_df.copy(), self.S_df.copy())

    def print_model(self):
        print("Metabolites:")
        print(self.m_df)
        print("Stoichiometries:")
        print(self.S_df)

    def __repr__(self):
        return f"LinearRespirationModel({self.m_df}, {self.S_df})"
    
    def _update_S(self):
        # last column is a note
        numeric_cols = self.S_df.columns[0:-3]
        # make a numpy array matrix
        self.S = self.S_df[numeric_cols].values.copy()

        # kcats have units mol C/mol enz/s
        self.kcat_s = self.S_df['kcat_s'].values
        # masses have units of kg/mol enz
        self.m_kDa = self.S_df['m_kDa'].values
        self.m_Da = self.m_kDa*1000
        self.NC = self.m_df.NC.values
        self.processes = self.S_df.index.values
        self.n_processes = self.S_df.index.size
        self.metabolites = self.m_df.index.values
        self.n_met = self.m_df.index.size

    def _check_c_balance(self):
        c_bal = np.round(self.S @ self.m_df.NC, decimals=1)
        if not (c_bal == 0).all():
            raise ValueError('C is not balanced {0}'.format(c_bal))
    
    def _check_e_balance(self):
        e_per_reactant = self.m_df.NC*self.m_df.NOSC
        e_balance = np.round(self.S @ e_per_reactant, decimals=1)
        
        if not (e_balance == 0).all():
            raise ValueError('e- are not balanced {0}'.format(e_balance))
    
    def get_S6(self):
        assert self.S_df.at['anabolism','EC'] == -self.S_df.at['anabolism','ECH']
        return self.S_df.at['anabolism','EC']

    def set_S6(self, new_S6):
        """Sets the S6 value and updates stoichiometries accordingly.
        
        Assumes that changes in S6 are due only to changing Z_C,B.
        """
        new_ZCB = self.ZCorg - 2*new_S6
        self.set_ZCB(new_ZCB)

    def set_ZCB(self, new_ZCB):
        """Sets the Z_C,B value and updates stoichiometries accordingly."""
        # S6 is the stoichiometric coefficient of the electron carrier 
        # in the anabolism reaction. We are updating that here. 
        new_S6 = (self.ZCorg - new_ZCB)/2
        # NOTE: electron carrier carries 2 electrons. 

        # Since all the fluxes are per-C, don't need to check stoichiometry.
        # Update the metabolite info - biomass is the only one that changes
        self.m_df.at['biomass', 'NOSC'] = new_ZCB
        self.ZCB = new_ZCB

        # update the stoichiometric matrix
        self.S_df.at['anabolism','EC'] = new_S6
        self.S_df.at['anabolism','ECH'] = -new_S6
        self._update_S()
        
        # check C and e- balancing
        self._check_c_balance()
        self._check_e_balance()

    def set_ATP_yield(self, process, new_yield):
        """Sets the ATP cost or yield of a process.
        
        Args:
            process: string name of process
            new_cost: float new ATP cost (negative) or yield (positive)
        """
        self.S_df.at[process, 'ATP'] = new_yield
        self.S_df.at[process, 'ADP'] = -new_yield
        
        self._update_S()
        self._check_c_balance()
        self._check_e_balance()

    def set_process_mass(self, process, new_mass):
        """Sets the mass of enzyme required for a process.

        Args:
            process: string name of process
            new_mass: float new mass in kDa
        """
        self.S_df.at[process, 'm_kDa'] = new_mass
        self._update_S()
        self._check_c_balance()
        self._check_e_balance()

    def set_process_masses(self, new_mass):
        """Sets the mass of enzyme required for all processes.

        Args:
            new_mass: float new mass in kDa
        """
        self.S_df.at[:, 'm_kDa'] = new_mass
        self._update_S()
        self._check_c_balance()
        self._check_e_balance()
    
    def set_process_kcat(self, process, new_kcat_s):
        """Sets the kcat of a process -- max rate of catalysis per active site.

        Args:
            process: string name of process
            new_kcat_s: float new kcat in mol C/mol E/s
        """
        self.S_df.at[process, 'kcat_s'] = new_kcat_s
        self._update_S()
        self._check_c_balance()
        self._check_e_balance()

    def max_growth_rate_problem(self, gr_opt_params):
        """Construct an LP maximizing the growth rate lambda.
        
        TODO: make the output problem DCP-compliant.
            https://www.cvxpy.org/tutorial/dcp/index.html#dcp

        Args:
            gr_opt_params: a GrowthRateOptimizationParams object.

        Returns:
            A cvxpy Problem object. Returned problem has a parameters:
                ks: per-process rate constant [mol C/mol E/s] per protein
                ms: per-process enzyme mass [g/mol E]
                phi_o: fraction of biomass allocated to "other" processes.
                max_lambda_hr: exponential growth rate in [1/hr] units (if max_lambda=True)
                fixed_ATP: fixed ATP concentration [KM units]
                fixed_NADH: fixed NADH concentration [KM units]
                maint: minimum maintenance ATP expenditure [mmol ATP/gDW/hr]
        """
        n_proc = self.n_processes
        n_met = self.n_met
        params = gr_opt_params

        constraints = []
        phis = cp.Variable(name='phis', shape=n_proc, nonneg=True)
        if params.phi_O is not None:
            my_phi_O = cp.Parameter(
                name='phi_O', value=params.phi_O, nonneg=True)
        elif params.min_phi_O is not None:
            min_phi_O_param = cp.Parameter(
                name='min_phi_O', value=params.min_phi_O, nonneg=True)
            my_phi_O = cp.Variable(name='phi_O', nonneg=True)
            constraints.append(my_phi_O >= min_phi_O_param)
        
        # Allocation constraint: sum(phi_i) = 1 by defn
        allocation_constr = cp.sum(phis) == 1-my_phi_O
        constraints.append(allocation_constr)

        # gamma = kcat / m on the assumption that yields are 1.
        # kcat_s: effective rate constant for process i, [mol C/mol E/s]
        # mcat_Da: per-process enzyme mass [g/mol E]
        # TODO: I think we can ignore NC since it's absorbed in the kcat. Check.
        gamma_vals = self.kcat_s / self.m_Da
        gammas = cp.Parameter(
            name='gammas', shape=n_proc, value=gamma_vals, pos=True)

        # Internal metabolites like ATP and NADH must have concentrations 
        # if we want to account for dilution and/or concentration-dependent fluxes.
        c_vals = np.zeros(n_met)
        ATP_index = self.m_df.index.get_loc('ATP')
        ADP_index = self.m_df.index.get_loc('ADP')
        NADH_index = self.m_df.index.get_loc('ECH')
        NAD_index = self.m_df.index.get_loc('EC')
        c_vals[ATP_index] = params.fixed_ATP
        c_vals[ADP_index] = params.fixed_ADP
        c_vals[NADH_index] = params.fixed_NADH
        c_vals[NAD_index] = params.fixed_NAD
        concs = cp.Parameter(
            name='concs', shape=n_met, nonneg=True, value=c_vals)

        # Calculate fluxes using on the rate law functor.
        # TODO: pass in the stoichioemtric matrix and metabolite names. 
        Js = params.rate_law.Apply(self.S, self.processes, self.metabolites,
                                   gammas, phis, concs)

        # Maximize the exponential growth rate by maximizing anabolic flux.
        # Though these are proportional, we convert units so opt has units of [1/hr].
        # J_ana has units of [number C/g/s], so we multiply by the mass of a C atom
        # and the ratio of grams cell per gram C. Finally we convert /s to /hr.
        # If we assume the biomass C fraction is fixed, this exactly equals the growth rate
        ana_idx = self.S_df.index.get_loc('anabolism')   
        growth_rate_s = Js[ana_idx]*GCELL_PER_GC*MW_C_ATOM     
        growth_rate_hr = growth_rate_s*S_PER_HR
        obj = cp.Maximize(growth_rate_hr)  # optimum has /hr units.

        # Maintenance is zero for all metabolites except ATP
        m_vals = np.zeros(n_met)
        m_vals[ATP_index] = params.ATP_maint
        m = cp.Parameter(name='maint', shape=n_met, nonneg=True, value=m_vals)

        # Flux balance constraint, including ATP maintenance and dilution if configured.
        metab_flux_balance = RHO_CELL*(self.S.T @ Js) - m
        if params.do_dilution:
            # Using growth rate in /s here to match units of fluxes.
            metab_flux_balance = (
                metab_flux_balance - cp.multiply(growth_rate_s, concs))

        # Can only enforce balancing for internal metabolites.
        internal_mets = self.m_df.internal.values.copy()
        internal_metab_flux_balance = cp.multiply(metab_flux_balance, internal_mets)
        constraints.append(internal_metab_flux_balance == 0)
        
        if params.max_lambda_hr:
            lambda_hr_ub = cp.Parameter(name='max_lambda_hr', nonneg=True,
                                        value=params.max_lambda_hr)
            constraints.append(growth_rate_hr <= lambda_hr_ub)

        # Construct the problem and return
        return cp.Problem(obj, constraints)
    
    def maximize_growth_rate(self, gr_opt_params):
        """Maximize growth rate at fixed phi_o.

        Args:
            gr_opt_params: a GrowthRateOptParams object.

        returns:
            two-tuple of (lambda, problem object). lambda = 0 when infeasible.
        """
        p = self.max_growth_rate_problem(gr_opt_params)
        soln = p.solve()
        if p.status in ("infeasible", "unbounded"):
            return 0, p
        
        # optimum has /hr units now.
        return p.value, p
    
    def model_as_dict(self):
        """Returns a dictionary of model parameters."""
        model_dict = defaultdict(float)
        for pname, k, m in zip(self.processes, self.kcat_s, self.m_kDa):
            model_dict[pname + '_kcat_s'] = k
            model_dict[pname + '_m_kDa'] = m
            model_dict[pname + '_gamma'] = k/(m*1000)          
        model_dict['ZCB'] = self.ZCB
        model_dict['ZCorg'] = self.ZCorg
        model_dict['ZCprod'] = self.ZCprod
        
        # Return the stoichiometries as positive values since this 
        # is assumed in the model definition and analytics.
        model_dict['S1'] = self.S_df.at['oxidation','ECH']
        model_dict['S2'] = self.S_df.at['reduction','EC']
        model_dict['S3'] = self.S_df.at['oxidation','ATP']
        model_dict['S4'] = self.S_df.at['reduction','ATP']
        model_dict['S5'] = self.S_df.at['anabolism','ADP']
        model_dict['S6'] = self.get_S6()
        
        return model_dict

    def solution_as_dict(self, opt_p):
        """Returns a dictionary of solution values for a solved problem."""
        soln_dict = defaultdict(float)
        has_opt = opt_p.status == 'optimal'
        opt_val = 0
        if has_opt:
            opt_val = opt_p.value
        max_lambda = opt_val

        soln_dict['lambda_hr'] = max_lambda
        soln_dict['maint'] = 0
        if 'maint' in opt_p.param_dict:
            soln_dict['maint'] = opt_p.param_dict['maint'].value
        if 'max_lambda_hr' in opt_p.param_dict:
            soln_dict['max_lambda_hr'] = opt_p.param_dict['max_lambda_hr'].value
        if 'phi_o' in opt_p.param_dict:
            soln_dict['phi_o'] = opt_p.param_dict['phi_o'].value
        elif 'phi_o' in opt_p.var_dict:
            soln_dict['phi_o'] = opt_p.var_dict['phi_o'].value
            soln_dict['min_phi_o'] = opt_p.param_dict['min_phi_o'].value

        gammas = opt_p.param_dict['gammas'].value
        phis = opt_p.var_dict['phis'].value
        if not has_opt:
            phis = np.zeros(self.n_processes)
        js = gammas*phis

        for pname, my_g, my_phi, my_j in zip(self.processes, gammas, phis, js):
            soln_dict[pname + '_gamma'] = my_g
            soln_dict[pname + '_phi'] = my_phi
            soln_dict[pname + '_flux'] = my_j

        return soln_dict

            
