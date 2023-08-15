import unittest

from linear_opt.lin_metabolism import GrowthRateOptParams, RateLawFunctor
from linear_opt.lin_metabolism import LinearMetabolicModel
from os import path


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
        self.assertEqual(opt.fixed_ATP, 0)
        self.assertEqual(opt.fixed_NADH, 0)

    def testMaintenance(self):
        opt = GrowthRateOptParams(maintenance_cost=1)
        self.assertEqual(opt.maintenance_cost, 1)

        # Double check unit conversion to molar/s
        # 1 mmol ATP/gDW/hr x 1e-3 mol/mmol x 0.3 gDW/g cell
        # x 1000 g cell / L cell x 1 hr / 3600 s = 1080 molar ATP/s
        self.assertAlmostEqual(opt.ATP_maint, 1080.0)
    
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

    def testModelAsDict(self):
        d = self.model.model_as_dict()
        for n in self.STOICHS + self.ZCS:
            self.assertTrue(n in d)

    def testMaxGrowthRateBasic(self):
        # Optimize with default params
        params = GrowthRateOptParams()
        optimum, problem = self.model.maximize_growth_rate(params)

        # Check the dictionary has some keys in it as expected.
        soln_dict = self.model.solution_as_dict(problem)
        process_names = self.model.S_df.index.values.tolist()
        for p in process_names:
            self.assertTrue(p + "_gamma" in soln_dict)
            self.assertTrue(p + "_phi" in soln_dict)
            self.assertTrue(p + "_flux" in soln_dict)

    def testMaxGrowthRateDilution(self):
        params = GrowthRateOptParams(
            do_dilution=True, fixed_ATP=0.01, fixed_NADH=0.01)
        optimum, problem = self.model.maximize_growth_rate(params)


if __name__ == '__main__':
    unittest.main()