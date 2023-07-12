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
    
    def __init__(self, metabolites_df, S_df, ):
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
    
    def max_anabolic_rate_problem(self, phi_o=0.001, fix_lambda=False, maint=None):
        """Construct an LP maximizing anabolic rate at fixed growth rate.
        
        Args:
            phi_o: fraction of biomass allocated to "other" processes.
                default value is intentionally unrealistically low.
            fix_lambda: if True, fixes the anabolic flux to a predetermined value.
                Useful for sweeping lambda values.
            maint: minimum maintenance ATP expenditure [mmol ATP/gDW/hr]
                if None, defaults to 0. Typical values 10-20 mmol ATP/gDW/hr

        Returned problem has a parameters:
            lambda_hr: exponential growth rate in [1/hr] units
            ks: per-process rate constant [mol C/mol E/s] per protein
            ms: per-process enzyme mass [g/mol E]
        """
        n_proc = self.n_processes
        n_met = self.n_met
        S = self.S

        # calculate per-process fluxes j_i, units of [mol C/gDW/s]
        # j_i = kcat_i * phi_i / m_i
        # phi_i: biomass fraction [unitless]
        # have a phi for each process, plus one for "other"
        phis = cp.Variable(name='phis', shape=n_proc, nonneg=True)
        phi_o = cp.Parameter(name='phi_o', value=phi_o, nonneg=True)

        # kcat: effective rate constant for process i, [mol C/mol E/s]
        ks = cp.Parameter(name='ks', shape=n_proc, value=self.kcat_s, pos=True)
        # m: per-process enzyme mass [g/mol E]
        ms = cp.Parameter(name='ms', shape=n_proc, value=self.m_Da, pos=True)

        # vector of fluxes j_i, note that the last phi is phi_o that has no rate law.
        js = cp.multiply(ks, phis) / ms

        # sum(phi_i) = 1 by defn
        allocation_constr = cp.sum(phis) == 1-phi_o

        # maximize growth rate by maximization of anabolic flux
        ana_idx = self.S_df.index.get_loc('anabolism')
        obj = cp.Maximize(js[ana_idx])
        
        # Biomass production must balance dilution by growth at fixed mu 
        # js are in units of mol C/gDW/s
        lambda_hr = cp.Parameter(name='lambda_hr', nonneg=True)

        # Though the anabolic flux is in C units, if we assume the biomass
        # carbon fraction is fixed, then it exactly equals the growth rate.        
        ana_flux = js[ana_idx]

        # Sets anabolic flux to a predetermined lambda_hr after converting to [1/s]
        per_s_conv = 1.0/3600
        fixed_lambda = ana_flux - lambda_hr*per_s_conv
        
        # Set up a minimum maintenance ATP expenditure constraint.
        # maintenance energy on ≈ 10-20 mmol ATP/gDW/hr (Hempfling & Maizner 1975)
        # converting into flux units [mol ATP/gDW/s] gives 1e-3*20/3600 = 5.6e-6
        # ^ double check above number
        min_ATP_consumption_hr = maint or 0
        min_ATP_consumption_s = 1e-3*min_ATP_consumption_hr/3600
        ATP_index = self.m_df.index.get_loc('ATP')

        # maintenance cost is uniformly zero for all metabolites other than ATP
        m_vals = np.zeros(n_met)
        m_vals[ATP_index] = min_ATP_consumption_s
        m = cp.Parameter(name='maint', shape=n_met, nonneg=True, value=m_vals)
        
        # Flux balance constraint, including ATP maintenance
        metab_flux_balance = S.T @ js
        metab_flux_balance = metab_flux_balance - m

        # Can only enforce balancing for internal metabolites.
        internal_mets = self.m_df.internal.values.copy()
        internal_metab_flux_balance = cp.multiply(metab_flux_balance, internal_mets)
        cons = [internal_metab_flux_balance == 0,
                allocation_constr]
        if fix_lambda:
            cons.append(fixed_lambda == 0)
        return cp.Problem(obj, cons)
    
    def maximize_lambda(self, phi_o=0.001):
        """Maximize growth rate at fixed phi_o.

        Args:
            phi_o: minimum fraction of biomass allocated to "other" processes.

        returns:
            two-tuple of (lambda, problem object). lambda = 0 when infeasible.
        """
        p = self.max_anabolic_rate_problem(phi_o=phi_o)
        soln = p.solve()
        if p.status in ("infeasible", "unbounded"):
            return 0, p
        
        lam_val = p.value*3600
        return lam_val, p

    def maximize_lambda_old(self, min_phi_o=0.001, lambda_min=0.05, lambda_max=5.16, lambda_step=0.1):
        """Maximize growth rate at fixed phi_o.

        Args:
            min_phi_o: minimum fraction of biomass allocated to "other" processes.
            lambda_min: minimum lambda to test
            lambda_max: maximum lambda to test
            lambda_step: step size for lambda maximization
        """
        increasing_lams = np.arange(lambda_min, lambda_max, lambda_step)
        lam_argmax = -1
        last_infeasible = False
        infeasible_ct = 0 
        p = self.max_anabolic_rate_problem(min_phi_o)
        
        # Increasing order of lambda [1/hr]
        for lam_val in increasing_lams:
            p.param_dict['lambda_hr'].value = lam_val
            soln = p.solve()
            if p.status in ("infeasible", "unbounded"):
                infeasible_ct += 1
                # Save a little computation by bailing if 
                # it seems like we have reached infeasible growth rates
                if infeasible_ct > 3 and last_infeasible:
                    break
                last_infeasible = True
                continue
            
            lam_argmax = lam_val
        
        # check sentinel
        if lam_argmax == -1:
            return None
            
        # second pass downward from an infeasible point (best mu + step size)
        # this way we can stop at the first feasible solution and 
        # return the maximal mu and the solved problem with solution state
        epsilon = lambda_step/10
        decreasing_lams = np.arange(lam_argmax+lambda_step, lam_argmax, -epsilon)
        for lam_val in decreasing_lams:
            p.param_dict['lambda_hr'].value = lam_val
            soln = p.solve()
            if p.status not in ("infeasible", "unbounded"):
                return lam_val, p
            
