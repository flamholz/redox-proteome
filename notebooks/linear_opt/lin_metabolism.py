import pandas as pd
import numpy as np
import cvxpy as cp

from collections import defaultdict

"""
Class uses resource balance/FBA style approach to optimize resource 
allocation in a simple 3 half reaction model of a single metabolism. 
"""

__author__ = "Avi I. Flamholz"


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
    
    def max_anabolic_rate_problem(self, phi_o=None,
                                  min_phi_o=None,
                                  max_lambda_hr=None,
                                  fixed_ATP=None,
                                  fixed_NADH=None,
                                  maint=None):
        """Construct an LP maximizing anabolic rate at fixed growth rate.

        Here we are assuming all processes are zero-order in substrate concentrations,
        i.e. saturated with substrate in the Michaelis-Menten sense. So if a fixed
        ATP or NADH concentration is given, we use it to account for dilution, but it 
        does not affect other fluxes.

        Args:
            phi_o: if not null, exact fraction of biomass allocated
                to "other" processes.
            min_phi_o: if not null, minimum fraction of biomass allocated
                to "other" processes.
            max_lambda_hr: if not None, sets the maximum exponential
                growth rate. In units of [1/hr].
            fixed_ATP: if not None, sets the ATP concentration to a fixed value.
            fixed_NADH: if not None, sets the NADH concentration to a fixed value.
            maint: minimum maintenance ATP expenditure [mmol ATP/gDW/hr]
                if None, defaults to 0. Typical values 10-20 mmol ATP/gDW/hr

        Returned problem has a parameters:
            ks: per-process rate constant [mol C/mol E/s] per protein
            ms: per-process enzyme mass [g/mol E]
            phi_o: fraction of biomass allocated to "other" processes.
            max_lambda_hr: exponential growth rate in [1/hr] units (if max_lambda=True)
            fixed_ATP: fixed ATP concentration [KM units]
            fixed_NADH: fixed NADH concentration [KM units]
            maint: minimum maintenance ATP expenditure [mmol ATP/gDW/hr]
        """
        assert phi_o is None or min_phi_o is None, "Can't set both phi_o and min_phi_o"

        n_proc = self.n_processes
        n_met = self.n_met

        constraints = []

        # calculate per-process fluxes j_i, units of [mol C/gDW/s]
        # j_i = kcat_i * phi_i / m_i
        # phi_i: biomass fraction [unitless]
        # have a phi for each process, plus one for "other"
        phis = cp.Variable(name='phis', shape=n_proc, nonneg=True)
        if phi_o is not None:
            phi_o = cp.Parameter(name='phi_o', value=phi_o, nonneg=True)
        elif min_phi_o is not None:
            min_phi_o_param = cp.Parameter(name='min_phi_o', value=min_phi_o, nonneg=True)
            phi_o = cp.Variable(name='phi_o', nonneg=True)
            constraints.append(phi_o >= min_phi_o_param)

        # kcat: effective rate constant for process i, [mol C/mol E/s]
        ks = cp.Parameter(name='ks', shape=n_proc, value=self.kcat_s, pos=True)
        # ms: per-process enzyme mass [g/mol E]
        ms = cp.Parameter(name='ms', shape=n_proc, value=self.m_Da, pos=True)
        # ns: per-metabolite carbon number [mol C/mol M]
        ncs = cp.Parameter(name='ncs', shape=n_met, value=self.NC, pos=True)

        # vector of fluxes j_i, note that the last phi is phi_o that has no rate law.
        js = cp.multiply(ks, phis) / ms

        # sum(phi_i) = 1 by defn
        allocation_constr = cp.sum(phis) == 1-phi_o
        constraints.append(allocation_constr)

        # Maximize the exponential growth rate by maximization of anabolic flux.
        # Though the anabolic flux is in C/s units, if we assume the biomass
        # carbon fraction is fixed, then it exactly equals the growth rate
        ana_idx = self.S_df.index.get_loc('anabolism')
        g_c_per_g_cell = 12*2
        s_per_hr = 3600
        growth_rate_hr = js[ana_idx]*s_per_hr*g_c_per_g_cell
        obj = cp.Maximize(growth_rate_hr)  # optimum has /hr units.
        
        # Set up a minimum maintenance ATP expenditure constraint.
        # maintenance energy on ≈ 10-20 mmol ATP/gDW/hr (Hempfling & Maizner 1975)
        # converting into flux units [mol ATP/gDW/s] gives 1e-3*20/3600 = 5.6e-6
        # ^ double check above number
        min_ATP_consumption_hr = maint or 0
        min_ATP_consumption_s = 1e-3*min_ATP_consumption_hr/3600
        ATP_index = self.m_df.index.get_loc('ATP')
        NADH_index = self.m_df.index.get_loc('ECH')

        # maintenance cost is uniformly zero for all metabolites other than ATP
        m_vals = np.zeros(n_met)
        m_vals[ATP_index] = min_ATP_consumption_s
        m = cp.Parameter(name='maint', shape=n_met, nonneg=True, value=m_vals)
        
        # only ATP and NADH can have concentrations, which are used only for dilution 
        c_vals = np.zeros(n_met)
        ATP_conc = fixed_ATP or 0
        NADH_conc = fixed_NADH or 0
        c_vals[ATP_index] = ATP_conc
        c_vals[NADH_index] = NADH_conc
        c = cp.Parameter(name='concs', shape=n_met, nonneg=True, value=c_vals)

        # Flux balance constraint, including ATP maintenance and dilution if needed.
        # Since J_ana = lambda, dilution term is -C*J_ana
        lam = js[ana_idx]/g_c_per_g_cell
        rho = 1000 # [g/L]
        metab_flux_balance = rho*(self.S.T @ js)/ncs - cp.multiply(lam, c) - m

        # Can only enforce balancing for internal metabolites.
        internal_mets = self.m_df.internal.values.copy()
        internal_metab_flux_balance = cp.multiply(metab_flux_balance, internal_mets)
        constraints.append(internal_metab_flux_balance == 0)
        
        if max_lambda_hr is not None:
            # We were told to optimize at fixed per-hr growth rate
            lambda_hr = cp.Parameter(name='max_lambda_hr', nonneg=True,
                                     value=max_lambda_hr)
            # Sets anabolic flux to given lambda_hr after converting to [1/s]
            constraints.append(growth_rate_hr <= lambda_hr)

        # Construct the problem and return
        return cp.Problem(obj, constraints)
    
    def maximize_lambda(self, phi_o=None,
                        min_phi_o=None,
                        max_lambda_hr=None,
                        fixed_ATP=None,
                        fixed_NADH=None,
                        maint=None):
        """Maximize growth rate at fixed phi_o.

        Args:
            phi_o: if not None, sets the exact fraction of biomass
                allocated to "other" processes.
            min_phi_o: if not None, sets the minimum fraction of biomass
                allocated to "other" processes.
            max_lambda_hr: if not None, sets the maximum exponential growth rate.
                In units of [1/hr].
            fixed_ATP: if not None, sets the ATP concentration to a fixed value.
            fixed_NADH: if not None, sets the NADH concentration to a fixed value.
            maint: minimum maintenance ATP expenditure [mmol ATP/gDW/hr]

        returns:
            two-tuple of (lambda, problem object). lambda = 0 when infeasible.
        """
        p = self.max_anabolic_rate_problem(
            phi_o=phi_o, min_phi_o=min_phi_o,
            max_lambda_hr=max_lambda_hr,
            fixed_ATP=fixed_ATP, fixed_NADH=fixed_NADH,
            maint=maint)
        soln = p.solve()
        if p.status in ("infeasible", "unbounded"):
            return 0, p
        
        # optimum has /hr units now.
        return p.value, p
    
    def model_as_dict(self):
        """Returns a dictionary of model parameters."""
        soln_dict = defaultdict(float)
        for pname, k, m in zip(self.processes, self.kcat_s, self.m_kDa):
            soln_dict[pname + '_kcat_s'] = k
            soln_dict[pname + '_m_kDa'] = m
            soln_dict[pname + '_gamma'] = k/(m*1000)          
        soln_dict['ZCB'] = self.ZCB
        soln_dict['ZCorg'] = self.ZCorg
        soln_dict['ZCprod'] = self.ZCprod
        
        # Return the stoichiometries as positive values since this 
        # is assumed in the model definition and analytics.
        soln_dict['S1'] = self.S_df.at['oxidation','ECH']
        soln_dict['S2'] = self.S_df.at['reduction','EC']
        soln_dict['S3'] = self.S_df.at['oxidation','ATP']
        soln_dict['S4'] = self.S_df.at['reduction','ATP']
        soln_dict['S5'] = self.S_df.at['anabolism','ADP']
        soln_dict['S6'] = self.get_S6()
        
        return soln_dict

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

        ks = opt_p.param_dict['ks'].value
        ms = opt_p.param_dict['ms'].value
        phis = opt_p.var_dict['phis'].value
        if not has_opt:
            phis = np.zeros(self.n_processes)
        js = ks*phis/ms

        for pname, my_phi, my_j in zip(self.processes, phis, js):
            soln_dict[pname + '_phi'] = my_phi
            soln_dict[pname + '_flux'] = my_j

        return soln_dict

            
