import unittest
import numpy as np

from linear_opt.lin_metabolism import GrowthRateOptParams, RateLawFunctor
from linear_opt.lin_metabolism import LinearMetabolicModel
from linear_opt.lin_metabolism import SingleSubstrateMMRateLaw
from linear_opt.lin_metabolism import MultiSubstrateMMRateLaw
from linear_opt.lin_metabolism import MW_C_ATOM

from os import path

# Approximate concentrations and ratios for plotting
# Based on Bennett et al. 2009 measurements in E. coli
DEFAULT_ATP = 1.4e-6
DEFAULT_NADH = 1.2e-7
DEFAULT_RE = 10
DEFAULT_RA = 0.3


class FakeRateLaw(RateLawFunctor):
    """Mock rate law functor with higher order."""
    ORDER = 10


class GrowthRateOptParamsTest(unittest.TestCase):
    """Test growth rate params."""

    def testEmptyParams(self):
        opt = GrowthRateOptParams()
        self.assertEqual(opt.rate_law.ORDER, 0)
        self.assertEqual(opt.min_phi_O, 0)
        self.assertEqual(opt.ATP_maint, 0)
        self.assertEqual(opt.fixed_ATP, 1)
        self.assertEqual(opt.fixed_NADH, 1)
        self.assertEqual(opt.fixed_ADP, 1)
        self.assertEqual(opt.fixed_NAD, 1)
        self.assertEqual(opt.fixed_C_red, 1)
        
    def testMaintenance(self):
        opt = GrowthRateOptParams(maintenance_cost=1)
        self.assertEqual(opt.maintenance_cost, 1)

        # Double check unit conversion to mol/gCDW/s
        # 1 mmol ATP/gDW/hr x 2 gDW/gCDW x 1e-3 mol/mmol x 1 hr / 3600 s = 1080 molar ATP/s
        self.assertAlmostEqual(opt.ATP_maint, 5.556e-7)
    
    def testBothPhiO(self):
        self.assertRaises(
            AssertionError, GrowthRateOptParams, phi_O=1, min_phi_O=1)
        
    def testDilutionNoConcs(self):
        self.assertRaises(
            AssertionError, GrowthRateOptParams, do_dilution=True)
        
    def testHigherOrderRateLawNoConcs(self):
        self.assertRaises(
            AssertionError, GrowthRateOptParams, rate_law=FakeRateLaw())


class BasicModelTest(unittest.TestCase):
    """Test growth rate params."""

    STOICHS = 'S1,S2,S3,S4,S5,S6'.split(',')
    ZCS = 'ZCorg,ZCprod,ZCB'.split(',')

    def setUp(self):
        # Loading the model of respiration just to exercise the code. 
        model_dir = '../models/linear/respiration/'
        m_fname = path.join(model_dir, 'glucose_resp_molecular_props.csv')
        S_fname = path.join(model_dir, 'glucose_resp_stoich_matrix.csv')
        self.model = LinearMetabolicModel.FromFiles(m_fname, S_fname)

        # These are the values in the default model file above (for now)
        self.expected_Svals = dict(S1=2, S2=-0.5, S3=0.5, S4=1, S5=-0.3, S6=0)

    def testSetS6(self):
        for v in np.arange(-3, 3):
            zcorg = self.model.ZCorg
            expected_zcb = zcorg + 2*v
            self.model.set_S6(v)
            self.assertEqual(self.model.get_S6(), v)
            self.assertEqual(self.model.ZCB, expected_zcb)

    def testSetZCorg(self):
        for v in np.arange(-3, 3):
            zcb = self.model.ZCB
            expected_S6 = (zcb - v)/2
            self.model.set_ZCorg(v)
            self.assertEqual(self.model.get_S6(), expected_S6)
            self.assertEqual(self.model.ZCorg, v)

    def testSetZCB(self):
        for v in np.arange(-3, 3):
            zcorg = self.model.ZCorg
            expected_S6 = (v - zcorg)/2
            self.model.set_ZCB(v)
            self.assertEqual(self.model.get_S6(), expected_S6)
            self.assertEqual(self.model.ZCB, v)

    def testModelAsDict(self):
        d = self.model.model_as_dict()
        for n in self.STOICHS + self.ZCS:
            self.assertTrue(n in d)
        
        for k, v in self.expected_Svals.items():
            self.assertEqual(d[k], v)

    def testSetProcessMass(self):
        process_names = self.model.S_df.index.values.tolist()
        for mass in np.arange(0, 10, 0.1):
            for process in process_names:
                self.model.set_process_mass(process, 1)
                self.assertEqual(self.model.get_process_mass(process), 1)

    def testSetProcessMasses(self):
        process_names = self.model.S_df.index.values.tolist()
        for mass in np.arange(0, 10, 0.1):
            self.model.set_process_masses(mass)
            for process in process_names:
                self.assertEqual(self.model.get_process_mass(process), mass)

    def testMaxGrowthRateBasic(self):
        # Optimize with default params
        params = GrowthRateOptParams()
        optimum, problem = self.model.maximize_growth_rate(params)

        # Check the dictionary has some keys in it as expected.
        soln_dict = self.model.results_as_dict(problem, params)

        process_names = self.model.S_df.index.values.tolist()
        for p in process_names:
            self.assertTrue(p + "_gamma" in soln_dict)
            self.assertTrue(p + "_phi" in soln_dict)
            self.assertTrue(p + "_flux" in soln_dict)

            # Check that the fluxes are consistent with the gammas and phis
            # Assumes linear rate law, which is the default
            g = soln_dict[p + "_gamma"]
            phi = soln_dict[p + "_phi"]
            flux = soln_dict[p + "_flux"]
            self.assertAlmostEqual(g*phi, flux)
        
        # Check that anabolism flux is consistent with the growth rate
        # Assumes linear rate law, which is the default
        J_ana = soln_dict['anabolism_flux']
        phi_ana = soln_dict['anabolism_phi']
        gamma_ana = soln_dict['anabolism_gamma']
        self.assertAlmostEqual(gamma_ana*phi_ana, J_ana)
        growth_rate = soln_dict['lambda_hr']
        mC = MW_C_ATOM
        self.assertAlmostEqual(3600*mC*J_ana, growth_rate)
        self.assertAlmostEqual(3600*mC*gamma_ana*phi_ana, growth_rate)
        self.assertEquals(optimum, growth_rate)
    
    def testMaxGrowthRateFirstOrder(self):
        # Optimize with default params
        rl = SingleSubstrateMMRateLaw()
        params = GrowthRateOptParams(rate_law=rl,
                                     fixed_ATP=0.01, fixed_NADH=0.01)
        optimum, problem = self.model.maximize_growth_rate(params)
    
    def testMaxGrowthRateFirstOrderConcRatios(self):
        # Optimize with default params
        rl = SingleSubstrateMMRateLaw()
        params = GrowthRateOptParams(
            rate_law=rl,
            fixed_ATP=0.01, fixed_ra=0.1,
            fixed_NADH=0.01, fixed_re=0.1)
        optimum, problem = self.model.maximize_growth_rate(params)

    def testMaxGrowthRateMultiSubstrate(self):
        rl = MultiSubstrateMMRateLaw()
        params = GrowthRateOptParams(rate_law=rl,
                                     fixed_ATP=0.01, fixed_NADH=0.01)
        optimum, problem = self.model.maximize_growth_rate(params)

    def testMaxGrowthRateMultiSubstrateConcRatios(self):
        rl = MultiSubstrateMMRateLaw()
        params = GrowthRateOptParams(
            rate_law=rl,
            fixed_ATP=0.01, fixed_ra=0.1,
            fixed_NADH=0.01, fixed_re=0.1)
        optimum, problem = self.model.maximize_growth_rate(params)

    def testMaxGrowthRateDilution(self):
        params = GrowthRateOptParams(
            do_dilution=True, fixed_ATP=0.01, fixed_NADH=0.01)
        optimum, problem = self.model.maximize_growth_rate(params)

    def testProblemAsDict(self):
        # Put some concentrations in there
        params = GrowthRateOptParams(do_dilution=True,
                                     fixed_ATP=0.01, fixed_NADH=0.01)
        optimum, problem = self.model.maximize_growth_rate(params)
        d = self.model.solution_as_dict(problem, params)

        # Check that the concs we put in are there
        self.assertEqual(d['ATP_conc'], 0.01)
        self.assertEqual(d['ECH_conc'], 0.01)


class AutoModelTest(unittest.TestCase):
    """Test growth rate params."""

    STOICHS = 'S1,S2,S3,S4,S5,S6'.split(',')
    ZCS = 'ZCorg,ZCprod,ZCB'.split(',')

    def setUp(self):
        # Loading the model of respiration just to exercise the code. 
        model_dir = '../models/linear/autotrophy/'
        m_fname = path.join(model_dir, 'glucose_auto_molecular_props.csv')
        S_fname = path.join(model_dir, 'glucose_auto_stoich_matrix.csv')
        self.model = LinearMetabolicModel.FromFiles(m_fname, S_fname)
    
    def testSetZCorg(self):
        for v in np.arange(-3, 3):
            zcb = self.model.ZCB
            expected_S6 = (zcb - v)/2
            self.model.set_ZCorg(v, heterotroph=False)
            self.assertEqual(self.model.get_S6(), expected_S6)
            self.assertEqual(self.model.ZCorg, v)

    def testSetZCB(self):
        for v in np.arange(-3, 3):
            zcorg = self.model.ZCorg
            expected_S6 = (v - zcorg)/2
            self.model.set_ZCB(v)
            self.assertEqual(self.model.get_S6(), expected_S6)
            self.assertEqual(self.model.ZCB, v)

if __name__ == '__main__':
    unittest.main()