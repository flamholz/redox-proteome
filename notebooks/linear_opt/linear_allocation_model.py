import pandas as pd
import numpy as np
import cvxpy as cp

from collections import defaultdict


class LinearAllocationModel(object):
    """Coarse-grained linear model of allocation of proteome resources while balancing ATP and e- fluxes."""
    
    def __init__(self, metabolites_df, S_df, ):
        self.m_df = metabolites_df
        self.S_df = S_df
        
        # Calculate the difference in NOSC between intermediates and protein
        self.dNOSC = self._calc_dNOSC()
        
        # make a numeric stoichiometric matrix
        self._update_S()
        # check C and e- balancing
        self._check_c_balance()
        self._check_e_balance()
    
    def _calc_dNOSC(self):
        # dNOSC is the delta between the intermediate and amino acids (precursor)
        int_NOSC = self.m_df.loc['intermediate'].NOSC
        pre_NOSC = self.m_df.loc['precursor'].NOSC
        prot_NOSC = self.m_df.loc['protein'].NOSC
        if pre_NOSC != prot_NOSC:
            raise ValueError(
                'precursor (NOSC = {0}) and protein (NOSC = {1}) should have the same NOSC value'.format(
                    pre_NOSC, prot_NOSC))
        return int_NOSC - pre_NOSC
    
    def set_dNOSC(self, dNOSC):
        int_NOSC = self.m_df.loc['intermediate'].NOSC
        int_NC = self.m_df.loc['intermediate'].NC
        
        pre_NC = self.m_df.loc['precursor'].NC
        pre_NOSC = int_NOSC - dNOSC
        
        # Need the C stoichiometry of anabolism to count e-. 
        # Signs of the stoichiometric coeffs denote production/consumption.
        int_consumed = self.S_df.at['anabolism','intermediate']
        pre_produced = self.S_df.at['anabolism','precursor']
        #print('{0} intermediates consumed {1}C each NOSC = {2}'.format(int_consumed, int_NC, int_NOSC))
        #print('{0} precursors produced {1}C each NOSC = {2}'.format(pre_produced, pre_NC, pre_NOSC))
                
        # calculate the number of ECH required to make the precursor
        # from the intermediate given their NOSC values
        int_etot = int_consumed*int_NOSC*int_NC
        pre_etot = pre_produced*pre_NC*pre_NOSC
        #print('{0} formal e- on {1} intermediate'.format(int_etot, int_NC))
        #print('{0} formal e- on {1} intermediate'.format(pre_etot, pre_NC))
        
        # if int_etot is more positive than pre_etot, we need to add e- to int to get pre. 
        # adding e- means that anabolism *consumes* NADH, so we subtract int_etot from pre_etot.
        # the number of carriers is half the difference since we are using a 2e- carrier.
        n_ech = (pre_etot + int_etot)/2.0
        #print('# NADH', n_ech)
        
        # update the metabolite info - precursor and protein have the same NOSC
        self.m_df.at['precursor', 'NOSC'] = pre_NOSC
        self.m_df.at['protein', 'NOSC'] = pre_NOSC
        self.dNOSC = dNOSC
        
        # update the stoichiometric matrix
        self.S_df.at['anabolism','EC'] = -n_ech
        self.S_df.at['anabolism','ECH'] = n_ech
        self._update_S()
        
        # check C and e- balancing
        self._check_c_balance()
        self._check_e_balance()
        
    def set_n_enzymes(self, n_enzymes):
        assert self.n_enz.size == len(n_enzymes)
        self.S_df['n_enzymes'] = n_enzymes
        self._update_S()
        
    def _update_S(self):
        # last column is a note
        numeric_cols = self.S_df.columns[0:-2]
        # make a numpy array matrix
        self.S = self.S_df[numeric_cols].values.copy()
        self.n_enz = self.S_df['n_enzymes'].values
        self.n_processes = self.S_df.index.size
        
        # v_i: rate constant [mol/s/g] per protein
        # default value set by assuming kcat ≈ 10 /s, MW = 30 kDa = 3e4 g/mol.
        # v_i = 10/3e4 = 3.33e-4 mol/g/s.
        default_v = 3e-4
        
        # vi are rescaled by the number of proteins involved so that costs (1/vi)
        # reflect the mass of protein needed to catalyze 1 flux unit
        self.vs = np.ones(self.n_processes)*default_v/self.n_enz
        
    def _check_c_balance(self):
        c_bal = np.round(self.S @ self.m_df.NC, decimals=1)
        if not (c_bal == 0).all():
            raise ValueError('C is not balanced {0}'.format(c_bal))
    
    def _check_e_balance(self):
        e_per_reactant = self.m_df.NC*self.m_df.NOSC
        e_balance = np.round(self.S @ e_per_reactant, decimals=1)
        
        # respiration is allowed to be imbalanced since it exchanges 
        # e- for ATP without an explicit terminal e- acceptor 
        resp_idx = self.S_df.index.get_loc('respiration')
        e_balance[resp_idx] = 0
        
        if not (e_balance == 0).all():
            raise ValueError('e- are not balanced {0}'.format(e_balance))
    
    def max_translation_rate_problem(self):
        """Construct an LP maximizing translation rate at fixed growth rate.
        
        Returned problem has a parameters:
            rho_prot: the protein density of cells [g/L]
            mu_hr: exponential growth rate in [1/hr] units
            vs: per-process rate constant [mol/s/g] per protein
        """
        n_proc = self.n_processes
        S = self.S

        # per-process fluxes j_i, units of mol/L cell volume/hour
        # j_i = v_i * phi_i * rho_prot
        phis = cp.Variable(name='phis', shape=n_proc, nonneg=True)

        # v_i: rate constant [mol/s/g] per protein, see self._update_S
        vs = cp.Parameter(
            name='vs', shape=n_proc, value=self.vs, pos=True)

        # rho_prot = M_prot/V_cell protein mass density ≈ 250 g/L (Milo 2013)
        rho_prot = cp.Parameter(name='rho_prot', value=250, pos=True)

        # fluxes j_i = rho_prot*v_i*phi_i [mol/L/s]
        # phi_i: proteome fraction [unitless]
        # rho_prot: protein mass density of cells, g/L
        # v_i: mass-specific rate constant, mol/g/s
        js = rho_prot*cp.multiply(vs, phis)

        # sum(phi_i) = 1 by defn
        allocation_constr = cp.sum(phis) == 1

        # maximize growth rate by maximization of translation flux
        # translation is the second-to-last process
        obj = cp.Maximize(js[-2])
        
        # Proteins production must balance dilution by growth at fixed mu 
        # js are in units of amino acid flux [mol/L/s]
        # rho_prot is a mass density [g/L]
        # to convert we divide by ≈100 g/mol amino acids to get [mol/L] units.
        # additionally, mu was provided in per-hour, but fluxes have per-s units.
        per_s_conv = 1.0/3600
        g_mol_aa = 100
        mu_hr = cp.Parameter(name='mu_hr', nonneg=True)
        
        # js[-2] is the total protein production rate, which must be
        # balanced by dilution due to growth at steady-state
        translation_flux = js[-2]
        protein_flux_balance = (
            translation_flux - mu_hr*per_s_conv*rho_prot/g_mol_aa)
        
        # I would prefer to balance the individual categories, but then
        # I need to calcualte the per-category production, which requires
        # me to multiply two variables: phi_i*translation_rate
        
        # phi_o catalyzes "homeostasis" flux which we take to represent maintenance
        # S matrix defines ATP consumption stoich of homeostasis (1 ATP -> ADP).
        # maintenance energy on glucose is ≈ 20 mmol ATP/gDW/hr (Hempfling & Maizner 1975)
        # converting to mmol ATP/g protein/S assuming 1 gDW contains 0.5 g protein,
        # then into flux units [mol ATP/L/s] where L is a liter of cell volume
        homeostasis_ATP_flux = js[-1]*self.S_df.at['homeostasis', 'ADP']  # ADP coeff is positive
        min_ATP_consumption_s = 1e-3*rho_prot*20*2/3600 # ≈ 0.003
        
        # add maintenance energy production constraint.
        maintenance_energy = homeostasis_ATP_flux >= min_ATP_consumption_s
                
        # Can only enforce balancing for internal metabolites, 
        # but this means ATP, e- and C will be balanced internally
        metab_flux_balance = S.T @ js
        internal_mets = self.m_df.internal.values.copy()
        internal_metab_flux_balance = cp.multiply(metab_flux_balance, internal_mets)
        cons = [internal_metab_flux_balance == 0,
                protein_flux_balance == 0,
                maintenance_energy,
                allocation_constr]
        return cp.Problem(obj, cons)
    
    def maximize_mu(self, mu_min=0.05, mu_max=5.16, mu_step=0.1):
        mus = np.arange(mu_min, mu_max, mu_step)
        mu_argmax = -1
        opt = None 
        p = self.max_translation_rate_problem()
        
        # Increasing order of mu [1/hr]
        for mu_val in mus:
            p.param_dict['mu_hr'].value = mu_val
            soln = p.solve()
            if p.status in ("infeasible", "unbounded"):
                continue
            
            mu_argmax = mu_val
            opt = soln
        
        # check sentinel
        if opt == None:
            return None
            
        # second pass downward from an infeasible point (best mu + step size)
        # this way we can stop at the first feasible solution and 
        # return the maximal mu and the solved problem with solution state
        epsilon = mu_step/10
        mus2 = np.arange(mu_argmax+mu_step, mu_argmax, -epsilon)
        for mu_val in mus2:
            p.param_dict['mu_hr'].value = mu_val
            soln = p.solve()
            if p.status not in ("infeasible", "unbounded"):
                return mu_val, p
            
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