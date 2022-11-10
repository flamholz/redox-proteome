import pandas as pd
import numpy as np
import cvxpy as cp


class LinearAllocationModel(object):
    """Coarse-grained linear model of allocation of proteome resources while balancing ATP and e- fluxes."""
    
    def __init__(self, metabolites_df, S_df):
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
        
        # calculate the number of ECH required to make the precursor
        # from the intermediate given their NOSC values
        int_etot = int_NOSC*int_NC
        pre_etot = pre_NC*pre_NOSC
        # number of carriers is half the number of e- since we are using a 2e- carrier
        n_ech = (pre_etot - int_etot)/2.0
        
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
        
    def _update_S(self):
        # last column is a note
        numeric_cols = self.S_df.columns[0:-2]
        # make a numpy array matrix
        self.S = self.S_df[numeric_cols].values.copy()
        self.n_enz = self.S_df['n_enzymes'].values
        
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
    
    def max_phi_r_problem(self):
        # 6 named process in the matrix, plus "other"
        n_processes = self.S_df.index.size
        S = self.S

        # per-process fluxes j_i, units of mol/L cell volume/hour
        # j_i = v_i * phi_i * rho_prot
        phis = cp.Variable(name='phis', shape=n_processes, nonneg=True)
        mu = cp.Parameter(name='mu', nonneg=True)

        # v_i: rate constant [mol/s/g] per protein
        # default value set by assuming kcat ≈ 10 /s, MW = 30 kDa = 3e4 g/mol.
        # v_i = 10/3e4 = 3.33e-4 mol/g/s.
        default_v = 3e-4
        # default vi are rescaled by the number of proteins involved so that
        # costs (1/vi) reflect the mass of protein needed to catalyze 1 flux unit
        default_vs = np.ones(n_processes)*default_v/self.n_enz
        vs = cp.Parameter(name='vs', shape=n_processes, 
                          value=default_vs, pos=True)

        # rho_prot = M_prot/V_cell protein mass density ≈ 250 g/L (Milo 2013)
        rho_prot = cp.Parameter(name='rho_prot', value=250, pos=True)

        # fluxes j_i = rho_prot*v_i*phi_i
        # phi_i: proteome fraction [unitless]
        js = rho_prot*cp.multiply(vs, phis)

        # sum(phi_i) = 1 by defn
        allocation_constr = cp.sum(phis) == 1

        # maximize growth rate by maximization of translation flux
        # translation is the second-to-last process
        obj = cp.Maximize(js[-2])

        # Can only enforce balancing for internal metabolites, 
        # but this means ATP, e- and C will be balanced internally
        internal_mets = self.m_df.internal.values.copy()
        flux_balance = S.T @ js
        internal_flux_balance = cp.multiply(flux_balance, internal_mets)
        cons = [internal_flux_balance  == 0, allocation_constr]
        return cp.Problem(obj, cons)
    
    @classmethod
    def FromFiles(cls, metabolites_fname, stoich_fname):
        m_df = pd.read_csv(metabolites_fname, index_col=0)
        S_df = pd.read_csv(stoich_fname, index_col=0)
        return cls(m_df, S_df)