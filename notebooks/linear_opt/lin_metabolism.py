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
MW_C_ATOM       = 12.0  # molar mass of a carbon atom [g/mol]
S_PER_HR        = 60*60 # seconds per hour 
GC_PER_GDW      = 0.5   # g carbon per g dry weight
# TODO: double check this value
GC_PER_GPROTEIN = 0.5   # g carbon per g protein


class RateLawFunctor(object):
    """Abstract base class for rate law functors."""

    ORDER = None
    NAME = None

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
    NAME = 'ZeroOrder'

    def Apply(self, S, processes, metabolites, gammas, phis, concs):
        return cp.multiply(gammas, phis)
    

class SingleSubstrateMMRateLaw(RateLawFunctor):
    """Michaelis-Menten type rate law.
    
    In this simple version, each flux depends on only one substrate concentration.

    Oxidation depends on NAD, reduction on NADH, anabolism and homeostasis on ATP.
    Concentrations are normalized by the KM value given on construction. 
    """
    ORDER = 1
    NAME = 'SingleSubstrateMM'

    def __init__(self, KM=6e-7):
        """Initializes the SimpleFirstOrderRateLaw class.

        Args:
            KM: float, Michaelis-Menten constant.
        """
        self.KM = KM

    def _rescale_concs(self, concs):
        """Saturating dependence on concentration."""
        normed = concs / self.KM
        res = normed / (1 + normed)
        return res

    def Apply(self, S, processes, metabolites, gammas, phis, concs):
        m_list = list(metabolites)
        p_list = list(processes)

        # In this simple version
        ATP_index = m_list.index('ATP')
        NADH_index = m_list.index('ECH')  # generic 2 e- carrier, reduced
        NAD_index = m_list.index('EC')    # generic 2 e- carrier, oxidized
        ana_index = p_list.index('anabolism')
        ox_index = p_list.index('oxidation')
        red_index = p_list.index('reduction')
        h_index = p_list.index('ATP_homeostasis')

        # binary matrix -- processes x metabolites -- indicating which metabolites
        # are substrates of which processes. Here each reaction can have only
        # one substrate.
        subs = np.zeros(S.shape)
        subs[ana_index, ATP_index] = 1
        subs[red_index, NADH_index] = 1
        subs[ox_index, NAD_index] = 1
        subs[h_index, ATP_index] = 1

        rescaled_concs = self._rescale_concs(concs) 
        conc_term = subs @ rescaled_concs
        return cp.multiply(gammas, cp.multiply(phis, conc_term))
            
    

class MultiSubstrateMMRateLaw(SingleSubstrateMMRateLaw):
    """First order rate law. 

    Fluxes depends on concentrations to the first power. In this version,
    each flux can depend on multiple substrate concentrations.

    Oxidation depends on NAD, reduction on NADH, anabolism on NADH and ATP.
    Concentrations are normalized by the KM value given on construction.
    """
    ORDER = 1
    NAME = 'MultiSubstrateMM'

    def __init__(self, KM=6e-7):
        """Initializes the SimpleFirstOrderRateLaw class.

        Args:
            KM: float, Michaelis-Menten constant.
                Default value is approx 1 mM in mol/gCDW units 
        """
        self.KM = KM

    def Apply(self, S, processes, metabolites, gammas, phis, concs):
        m_list = list(metabolites)
        p_list = list(processes)

        # Get the indices of the metabolites to make substrate matrices
        ATP_index = m_list.index('ATP')
        NADH_index = m_list.index('ECH')  # generic 2 e- carrier, reduced
        NAD_index = m_list.index('EC')    # generic 2 e- carrier, oxidized
        c_red_index = m_list.index('C_red')
        ec_ox_index = m_list.index('E_ox')
        biomass_index = m_list.index('biomass') 
        ana_index = p_list.index('anabolism')
        ox_index = p_list.index('oxidation')
        red_index = p_list.index('reduction')
        h_index = p_list.index('ATP_homeostasis')

        # binary matrix -- processes x metabolites -- indicating which metabolites are
        # substrates of which processes. Every reaction has to have exactly 1 substrate
        # in the matrix, otherwise we will be multiplying by zero (no substrate) or 
        # \sum(1*rescaled_conc) rather than 1*scaled_conc (the correct value).
        subs1 = np.zeros(S.shape)
        subs1[ana_index, ATP_index] = 1
        subs1[ox_index, NAD_index] = 1
        subs1[red_index, NADH_index] = 1
        subs1[h_index, ATP_index] = 1

        # second matrix for second set of substrates. here we are playing a little trick
        # we know that biomass always has a dummy concentration of 1, so we can use that
        # for homeostasis, which only has one substrate.
        subs2 = np.zeros(S.shape)
        subs2[ana_index, c_red_index] = 1
        subs2[ox_index, NADH_index] = 1
        subs2[red_index, ec_ox_index] = 1
        subs1[h_index, biomass_index] = 1

        scaled_concs = self._rescale_concs(concs)
        conc_term = cp.multiply(subs1 @ scaled_concs, subs2 @ scaled_concs)
        return cp.multiply(gammas, cp.multiply(phis, conc_term))
    

class GrowthRateOptParams(object):
    """A class to hold the parameters for the growth rate optimization problem.

    Attributes:
        do_dilution: boolean, whether to include dilution in the model.
        do_maintenance: boolean, whether to include maintenance in the model.
        rate_law: RateLawFunctor, rate law to use.
        min_phi_O: float, minimum C mass fraction for other processes.
        phi_O: float, mass C fraction for other processes.
            Only one of min_phi_O and phi_O should be set.
        max_phi_H: float, minimum C mass fraction for homeostasis.
        maintenance_cost: float, maintenance cost. Units of [mmol ATP/gDW/hr].
                These are typically reported units for convenience.
        ATP_maint: float, maintenance in units of [mol ATP/gCDW/s].
        max_lambda_hr: float, maximum lambda value. Units of [1/hr].
        max_C_uptake: float, maximum C uptake rate. Units of [mol C/gCDW/s].
        fixed_ATP: float, fixed ATP concentration. [mol/gCDW] units.
        fixed_NADH: float, fixed NADH concentration. [mol/gCDW] units.
        fixed_re: float, fixed ratio of NAD/NADH concentrations.
        fixed_ra: float, fixed ratio of ADP/ATP concentrations.
        fixed_NAD: float, fixed NAD concentration. [mol/gCDW] units.
        fixed_ADP: float, fixed ADP concentration. [mol/gCDW] units.
        fixed_C_red: float, fixed reduced C concentration. [mol/gCDW] units.
    """
    def __init__(self, do_dilution=False, rate_law=None,
                 min_phi_O=None, phi_O=None,
                 phi_red=None, max_phi_H=None,
                 maintenance_cost=0, max_lambda_hr=None, max_C_uptake=None,
                 fixed_ATP=None, fixed_NADH=None,
                 fixed_ra=None, fixed_re=None, fixed_C_red=None):
        """Initializes the GrowthRateOptParams class.

        Args:
            do_dilution: boolean, whether to include dilution in the model.
            rate_law: RateLawFunctor, rate law to use. If None, uses zeroth order.
            min_phi_O: float, minimum C mass fraction for other processes.
            phi_O: float, C mass fraction for other processes.
                Only one of min_phi_O and phi_O should be set.
            phi_red: float, C mass fraction catalyzing reduction.
            max_phi_H: float, maximum C mass fraction for homeostasis.
            maintenance_cost: float, maintenance cost. Units of [mmol ATP/gDW/hr].
                These are typically reported units for convenience.
            max_lambda_hr: float, maximum lambda value. Units of [1/hr].
            fixed_ATP: float, fixed ATP concentration. [mol/gCDW] units.
            fixed_NADH: float, fixed NADH concentration. [mol/gCDW] units.
            fixed_re: float, fixed ratio of NAD/NADH concentrations.
            fixed_ra: float, fixed ratio of ADP/ATP concentrations.
            fixed_C_red: float, fixed reduced C concentration. [mol/gCDW] units.
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
        self.phi_red = phi_red
        self.max_phi_H = max_phi_H
        self.max_lambda_hr = max_lambda_hr
        self.max_C_uptake = max_C_uptake

        # default concentrations are 1 so that
        # nothing changes if not specified.
        self.fixed_ATP = fixed_ATP or 1
        self.fixed_ra = fixed_ra or 1
        self.fixed_ADP = self.fixed_ATP * self.fixed_ra
        self.fixed_NADH = fixed_NADH or 1
        self.fixed_re = fixed_re or 1
        self.fixed_NAD = self.fixed_NADH * self.fixed_re
        self.fixed_C_red = fixed_C_red or 1

        # Convert maintenance to mol ATP/gCDW/s
        self.maintenance_cost = maintenance_cost or 0
        m = 1e-3*(self.maintenance_cost)/S_PER_HR   # mol ATP/gDW/s
        self.ATP_maint = m / GC_PER_GDW

    def as_dict(self):
        """Returns a dictionary of parameters."""
        max_phi_H = None
        if self.max_phi_H is not None:
            max_phi_H = self.max_phi_H

        return {
            'opt.do_dilution': self.do_dilution,
            'opt.min_phi_O': self.min_phi_O,
            'opt.phi_O': self.phi_O,
            'opt.max_phi_H_set': self.max_phi_H is not None,
            'opt.max_phi_H_value': self.max_phi_H,
            'opt.maintenance_cost_mmol_gDW_hr': self.maintenance_cost,
            'opt.ATP_maint_mol_gCDW_s': self.ATP_maint,
            'opt.max_lambda_hr': self.max_lambda_hr,
            'opt.max_C_uptake': self.max_C_uptake,
            'opt.fixed_ATP_mol_gCDW': self.fixed_ATP,
            'opt.fixed_NADH_mol_gCDW': self.fixed_NADH,
            'opt.fixed_NAD_mol_gCDW': self.fixed_NAD,
            'opt.fixed_ADP_mol_gCDW': self.fixed_ADP,
            'opt.fixed_C_red_mol_gCDW': self.fixed_C_red,
            'opt.fixed_ra': self.fixed_ra,
            'opt.fixed_re': self.fixed_re,
            'opt.rate_law_name': self.rate_law.NAME,
            'opt.rate_law_order': self.rate_law.ORDER,
        }
    
    def copy(self):
        return GrowthRateOptParams(
            do_dilution=self.do_dilution,
            rate_law=self.rate_law,
            min_phi_O=self.min_phi_O,
            phi_O=self.phi_O,
            max_phi_H=self.max_phi_H,
            maintenance_cost=self.maintenance_cost,
            max_lambda_hr=self.max_lambda_hr,
            max_C_uptake=self.max_C_uptake,
            fixed_ATP=self.fixed_ATP,
            fixed_NADH=self.fixed_NADH,
            fixed_ra=self.fixed_ra,
            fixed_re=self.fixed_re,
            fixed_C_red=self.fixed_C_red)

class LinearMetabolicModel(object):
    """Coarse-grained linear model of resource allocation while balancing ATP and e- fluxes."""

    @classmethod
    def FromFiles(cls, metabolites_fname, stoich_fname, heterotroph=True):
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
        return cls(m_df, S_df.astype(dtype_dict), heterotroph=heterotroph)
    
    def __init__(self, metabolites_df, S_df, heterotroph=True):
        self.m_df = metabolites_df
        self.S_df = S_df

        # boolean -- true if heterotroph, false if autotroph
        # indicates that Cred comes from outside the cell if true.
        self.heterotroph = heterotroph
        
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
        return LinearMetabolicModel(self.m_df.copy(), self.S_df.copy(),
                                    self.heterotroph)

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
        return self.S_df.at['anabolism','ECH']

    def set_S6(self, new_S6):
        """Sets the S6 value and updates stoichiometries accordingly.

        Note: consuming NADH, i.e. making reduced biomass, is negative S6.

        Assumes that changes in S6 are due only to changing Z_C,B.
        """
        new_ZCB = (self.ZCorg + 2*new_S6)
        self.set_ZCB(new_ZCB)

    def set_ZCorg(self, new_ZCorg):
        """Sets the Z_C,org value and updates stoichiometries accordingly.
        
        Assumes ZCB is fixed, recalculates S1/S2 and S6 accordingly.

        Args:
            new_ZCorg: float new Z_C,org value
        """
        new_SX = (self.ZCprod - new_ZCorg)/2
        new_S6 = (self.ZCB - new_ZCorg)/2

        # NOTE: electron carrier carries 2 electrons. 

        # Update the NOSC of reduced carbon 
        self.m_df.at['C_red', 'NOSC'] = new_ZCorg
        self.ZCorg = new_ZCorg

        # update the stoichiometric matrix
        self.S_df.at['anabolism','EC'] = -new_S6
        self.S_df.at['anabolism','ECH'] = new_S6

        if self.heterotroph:
            self.S_df.at['oxidation','EC'] = -new_SX
            self.S_df.at['oxidation','ECH'] = new_SX
        else:
            self.S_df.at['reduction','EC'] = new_SX
            self.S_df.at['reduction','ECH'] = -new_SX
        self._update_S()
        
        # check C and e- balancing
        self._check_c_balance()
        self._check_e_balance()

    def set_ZCB(self, new_ZCB):
        """Sets the Z_C,B value and updates stoichiometries accordingly.
        
        Assumes ZCorg is fixed, recalculates S6 accordingly.
        """
        # S6 is the number of electron carriers produced in anabolism. 
        # So it's negative when anabolism consumes NADH. 
        new_S6 = (new_ZCB - self.ZCorg)/2
        # NOTE: electron carrier carries 2 electrons. 

        # Since all the fluxes are per-C, don't need to check stoichiometry.
        # Update the metabolite info - biomass is the only one that changes
        self.m_df.at['biomass', 'NOSC'] = new_ZCB
        self.ZCB = new_ZCB

        # update the stoichiometric matrix
        self.S_df.at['anabolism','EC'] = -new_S6
        self.S_df.at['anabolism','ECH'] = new_S6
        self._update_S()
        
        # check C and e- balancing
        self._check_c_balance()
        self._check_e_balance()

    def get_ATP_yield(self, process):
        """Returns the ATP cost or yield of a process."""
        return self.S_df.at[process, 'ATP']

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

    def get_process_mass(self, process):
        """Returns the mass of enzyme required for a process in kDa."""
        return self.S_df.at[process, 'm_kDa']

    def set_process_masses(self, new_mass):
        """Sets the mass of enzyme required for all processes.

        Args:
            new_mass: float new mass in kDa
        """
        self.S_df['m_kDa'] = new_mass
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
        TODO: what should kcat be? currently 50 /s for all processes. 
        TODO: dilution for ADP and NAD? seems like I should.

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
        # mcat_Da: per-process enzyme mass [g/mol E]. Converting here to [gC/mol E]
        gamma_vals = self.kcat_s / (self.m_Da * GC_PER_GPROTEIN)
        gammas = cp.Parameter(
            name='gammas', shape=n_proc, value=gamma_vals, pos=True)

        # Internal metabolites like ATP and NADH must have concentrations 
        # if we want to account for dilution and/or concentration-dependent fluxes.
        c_vals = np.ones(n_met)
        ATP_index = self.m_df.index.get_loc('ATP')
        ADP_index = self.m_df.index.get_loc('ADP')
        NADH_index = self.m_df.index.get_loc('ECH')
        NAD_index = self.m_df.index.get_loc('EC')
        C_red_index = self.m_df.index.get_loc('C_red')
        c_vals[ATP_index] = params.fixed_ATP
        c_vals[ADP_index] = params.fixed_ADP
        c_vals[NADH_index] = params.fixed_NADH
        c_vals[NAD_index] = params.fixed_NAD
        c_vals[C_red_index] = params.fixed_C_red
        concs = cp.Parameter(
            name='concs', shape=n_met, nonneg=True, value=c_vals)

        # Calculate fluxes using on the rate law functor.
        # TODO: pass in the stoichioemtric matrix and metabolite names. 
        Js = params.rate_law.Apply(
            self.S, self.processes, self.metabolites,
            gammas, phis, concs)

        # Maximize the exponential growth rate by maximizing anabolic flux.
        # Though these are proportional, we convert units so opt has units of [1/hr].
        # J_ana has units of [number C/g/s], so we multiply by the mass of a C atom
        # and the ratio of grams cell per gram C. Finally we convert /s to /hr.
        # If we assume the biomass C fraction is fixed, this exactly equals the growth rate
        ana_idx = self.S_df.index.get_loc('anabolism')
        ox_index = self.S_df.index.get_loc('oxidation')
        growth_rate_s = Js[ana_idx]*MW_C_ATOM
        growth_rate_hr = growth_rate_s*S_PER_HR
        obj = cp.Maximize(growth_rate_hr)  # optimum has /hr units.

        # Maintenance is zero for all metabolites except ATP
        m_vals = np.zeros(n_met)
        m_vals[ATP_index] = params.ATP_maint
        m = cp.Parameter(name='maint', shape=n_met, nonneg=True, value=m_vals)

        # Flux balance constraint, including ATP maintenance and dilution if configured.
        metab_flux_balance = (self.S.T @ Js) - m
        if params.do_dilution:
            # Using growth rate in /s here to match units of fluxes.
            metab_flux_balance = (
                metab_flux_balance - cp.multiply(growth_rate_s, concs))

        # Can only enforce balancing for internal metabolites.
        internal_mets = self.m_df.internal.values.copy()
        internal_metab_flux_balance = cp.multiply(metab_flux_balance, internal_mets)
        constraints.append(internal_metab_flux_balance == 0)
        
        if params.max_lambda_hr is not None:
            lambda_hr_ub = cp.Parameter(name='max_lambda_hr', nonneg=True,
                                        value=params.max_lambda_hr)
            constraints.append(growth_rate_hr <= lambda_hr_ub)
        if params.max_C_uptake is not None:
            assert self.heterotroph, "Can only set max C uptake for heterotrophs."
            C_uptake_ub = cp.Parameter(name='max_C_uptake', nonneg=True,
                                       value=params.max_C_uptake)
            C_uptake_flux = Js[ana_idx] + Js[ox_index]
            constraints.append(C_uptake_flux <= C_uptake_ub)
        if params.max_phi_H is not None:
            h_index = self.S_df.index.get_loc('ATP_homeostasis')
            constraints.append(phis[h_index] <= params.max_phi_H)
        if params.phi_red is not None:
            # Fix the phi_red value
            red_index = self.S_df.index.get_loc('reduction')
            constraints.append(phis[red_index] == params.phi_red)

        # Construct the problem and return
        return cp.Problem(obj, constraints)
    
    def maximize_growth_rate(self, gr_opt_params, solver='GUROBI'):
        """Maximize growth rate at fixed phi_o.

        Args:
            gr_opt_params: a GrowthRateOptParams object.

        returns:
            two-tuple of (lambda, problem object). lambda = 0 when infeasible.
        """
        p = self.max_growth_rate_problem(gr_opt_params)
        soln = p.solve(solver=solver)
        #try:
        #    soln = p.solve()
        #except cp.SolverError:
        #    return 0, p
        
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
        model_dict['ZCB'] = self.ZCB
        model_dict['ZCorg'] = self.ZCorg
        model_dict['ZCprod'] = self.ZCprod
        
        # Return the stoichiometries from the matrix
        model_dict['S1'] = self.S_df.at['oxidation','ECH']
        model_dict['S2'] = self.S_df.at['reduction','ECH']
        model_dict['S3'] = self.S_df.at['oxidation','ATP']
        model_dict['S4'] = self.S_df.at['reduction','ATP']
        model_dict['S5'] = self.S_df.at['anabolism','ATP']
        model_dict['S6'] = self.get_S6()
        
        return model_dict

    def _analytics_zo(self, soln_dict):
        """Analytic values for zeroth order rate law.
        
        Returns:
            four tuple of the following variables
                lambda: growth rate calculated from fluxes
                lambda_max: maximum growth calculated from model parameters
                S6_lb: lower bound on viable S6 in the absence of ATP homeostasis
                S6_ub: upper bound on viable S6 in the absence of ATP homeostasis
        """
        sd = soln_dict
        g_ana = sd['anabolism_gamma']
        g_red = sd['reduction_gamma']
        g_ox = sd['oxidation_gamma']
        g_h = sd['ATP_homeostasis_gamma']
        phi_O = sd['phi_O']
        phi_H = sd['ATP_homeostasis_phi']
        b = sd['maint']
        A = sd['ATP_conc']
        N = sd['ECH_conc']
        mC = MW_C_ATOM
        S1, S2, S3 = sd['S1'], sd['S2'], sd['S3'], 
        S4, S5, S6 = sd['S4'], sd['S5'], sd['S6']

        # S6 bounds
        # Handle lower bound first
        phi_term = (phi_O-1)
        b_term = (b + S5*g_ox*phi_term)
        num =  S2*g_red*b_term + mC*g_ana*(b*N - (A*S2-N*S4)*phi_term*g_red)
        denom = g_ana*(b + S4*g_red*phi_term)
        S6_lb = num / denom 

        # Now upper bound
        num = S1*g_ox*b_term + mC*g_ana*(b*N - (A*S1-N*S3)*phi_term*g_red)
        denom = g_ana*(b + S3*g_ox*phi_term)
        S6_ub = num / denom

        # ZCorg bounds - lower bound first
        zcb = sd['ZCB']
        zcprod = sd['ZCprod']
        phi_term = (-1 + phi_H + phi_O)
        num2 = A*S2*g_red*phi_term
        denom2 = (b + g_h*phi_H + S4*g_red*phi_term)
        num1 = -2*S2*g_red*(b + g_h*phi_H + S5*g_ana*phi_term)
        denom1 = g_ana*denom2
        ZCorg_lb = (num1/denom1) + 2*mC*(-N + num2/denom2) + zcb
        
        # Now upper bound
        num = g_ana*(b + g_h*phi_H + S3*g_ox*phi_term)*(2*N*mC - zcb)
        num += g_ox*zcprod*(b + g_h*phi_H + A*mC*g_ana*phi_term + S5*g_ana*phi_term)
        denom = g_ox*(b + g_h*phi_H) + g_ana*(-b - g_h*phi_H - (A*mC + S3 - S5)*g_ox*phi_term)
        ZCorg_ub = num / denom

        # lambda calculated from interdependence of fluxes
        J_ox = sd['oxidation_flux']
        J_h = sd['ATP_homeostasis_flux']
        num = mC*(S2*(b + J_h - J_ox*S3)+J_ox*S1*S4)
        denom = mC*(A*S2-N*S4)-S2*S5+S4*S6
        lam = -3600*num/denom

        # maximum lambda from model parameters by Lagrange multipliers
        phi_term = (1 - phi_H - phi_O)
        S12_term = (S2*g_red-S1*g_ox)
        S34_term = (S4*g_red-S3*g_ox)
        num = S12_term*(S3*g_ox*phi_term - g_h*phi_H - b)
        num -= S1*g_ox*S34_term*phi_term
        denom = ((S5/mC) - A - (S3*g_ox)/(mC*g_ana))*S12_term
        denom -= ((S6/mC) - N - (S1*g_ox)/(mC*g_ana))*S34_term
        lam_max = -3600*num/denom

        return lam, lam_max, S6_lb, S6_ub, ZCorg_lb, ZCorg_ub

    def solution_as_dict(self, optimized_p, params):
        """Returns a dictionary of solution values for a solved problem.
        
        Args:
            optimized_p: a solved cvxpy Problem object.
            params: a GrowthRateOptParams object.
        
        Returns:
            A dictionary of solution values.
        """
        opt_p = optimized_p
        rl = params.rate_law

        soln_dict = defaultdict(float)
        soln_dict['opt_status'] = opt_p.status
        has_opt = opt_p.status == cp.OPTIMAL
        opt_val = 0
        if has_opt:
            opt_val = opt_p.value
        max_lambda = opt_val

        soln_dict['lambda_hr'] = max_lambda
        soln_dict['maint'] = 0
        if 'maint' in opt_p.param_dict:
            ATP_index = self.m_df.index.get_loc('ATP')
            soln_dict['maint'] = opt_p.param_dict['maint'].value[ATP_index]
        if 'max_lambda_hr' in opt_p.param_dict:
            soln_dict['max_lambda_hr'] = opt_p.param_dict['max_lambda_hr'].value
        if 'phi_O' in opt_p.param_dict:
            soln_dict['phi_O'] = opt_p.param_dict['phi_O'].value
        elif 'phi_O' in opt_p.var_dict:
            soln_dict['phi_O'] = opt_p.var_dict['phi_O'].value
            soln_dict['min_phi_O'] = opt_p.param_dict['min_phi_O'].value

        gammas = opt_p.param_dict['gammas'].value
        phis = opt_p.var_dict['phis'].value
        if not has_opt:
            phis = np.zeros(self.n_processes)

        # Get concs if they are in the model
        concs = None
        if 'concs' in opt_p.param_dict:
            met_names = self.m_df.index.values
            concs = opt_p.param_dict['concs'].value
            for m, c in zip(met_names, concs):
                soln_dict[m + '_conc'] = c
        
        # Get the fluxes using the rate law
        js = params.rate_law.Apply(
            self.S, self.processes, self.metabolites,
            gammas, phis, concs).value

        for pname, my_g, my_phi, my_j in zip(self.processes, gammas, phis, js):
            soln_dict[pname + '_gamma'] = my_g
            soln_dict[pname + '_phi'] = my_phi
            soln_dict[pname + '_flux'] = my_j

        return soln_dict
    
    def results_as_dict(self, optimized_p, params):
        """Returns a full dictionary of model and solution parameters."""
        d = self.model_as_dict()
        d.update(self.solution_as_dict(optimized_p, params))
        d.update(params.as_dict())

        # Add analytic vals after we put the model and solution in.
        # Note: we rely on the optimization to populate a few values
        # used in the analytic calculation, so success is required.
        #if optimized_p.status == cp.OPTIMAL:
        #    lam, lam_max, S6_lb, S6_ub, ZCorg_lb, ZCorg_ub = self._analytics_zo(d)
        #else:
        #    lam, lam_max, S6_lb, S6_ub, ZCorg_lb, ZCorg_ub = (
        #        np.NAN, np.NAN, np.NAN, np.NAN, np.NAN, np.NAN)
        lam, lam_max, S6_lb, S6_ub, ZCorg_lb, ZCorg_ub = self._analytics_zo(d)
        d['analytic_lambda_zo'] = lam
        d['analytic_lambda_max_zo'] = lam_max
        d['S6_lb_zo'] = S6_lb
        d['S6_ub_zo'] = S6_ub
        d['ZCorg_lb_zo'] = ZCorg_lb
        d['ZCorg_ub_zo'] = ZCorg_ub

        # Calculate carbon use efficiency, CUE
        # Since Jana and Jox are written per C, can just divide.
        Jana = d['anabolism_flux']
        Jox = d['oxidation_flux']
        Jsum = Jana + Jox
        d['CUE'] = np.NaN
        if Jsum > 0:
            d['CUE'] = Jana / Jsum
        return d
            
