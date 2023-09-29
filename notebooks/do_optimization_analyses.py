import numpy as np
import pandas as pd
import seaborn as sns
import viz

from linear_opt.lin_metabolism import MW_C_ATOM
from linear_opt.lin_metabolism import LinearMetabolicModel
from linear_opt.lin_metabolism import GrowthRateOptParams
from matplotlib import pyplot as plt
from os import path

from linear_opt.lin_metabolism import MW_C_ATOM
from linear_opt.lin_metabolism import LinearMetabolicModel
from linear_opt.lin_metabolism import GrowthRateOptParams

"""This script runs all model optimization analyses, 
saving results to CSV files for later plotting. 
"""

# Approximate concentrations and ratios for plotting
# Based on Bennett et al. 2009 measurements in E. coli
DEFAULT_ATP = 1.4e-6
DEFAULT_NADH = 1.2e-7
DEFAULT_RE = 10
DEFAULT_RA = 0.3

print('Loading respiration model...')
model_dir = '../models/linear/respiration/'
m_fname = path.join(model_dir, 'glucose_resp_molecular_props.csv')
S_fname = path.join(model_dir, 'glucose_resp_stoich_matrix.csv')
lam = LinearMetabolicModel.FromFiles(m_fname, S_fname)

print('Optimizing over a range of fixed lambda values...')
# Here is a model that has no homeostasis (phi_H <= 0),
# no maintenance (ATP_maint = 0), but can alter biomass
# composition through phi_O. We set a maximum lambda
# so that we can run the model over a range of lambda values.
lambdas = np.arange(0.1, 4, 0.01)
results = []

for lam_val in lambdas:
    # Make fresh parameters with a new max_lambda_hr
    params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, 
                                 max_lambda_hr=lam_val, max_phi_H=0,
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    p = params.copy()
    p.max_lambda_hr = lam_val

    # Optimize the growth rate given the parameters
    opt, opt_prob = lam.maximize_growth_rate(params)

    d = lam.results_as_dict(opt_prob, params)
    results.append(d)

phi_df = pd.DataFrame(results)
phi_df['expected_Jana'] = phi_df['anabolism_gamma']*phi_df['anabolism_phi']
phi_df['expected_lambda'] = MW_C_ATOM*3600*phi_df['expected_Jana']
phi_df.to_csv('../output/Fig2A_variable_lambda.csv')

print('Optimizing over a range of fixed Z_C,B values...')
# Sweep a range of biomass ZC values
ZCBs = np.arange(-3, 3.01, 0.05)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for z in ZCBs:
    # Test with and without ATP homeostasis -- first with
    ref_lam = lmm.copy()
    ref_lam.set_ZCB(z)
    # Note we are fixing phi_O here to highlight the contribution of phi_H
    params = GrowthRateOptParams(phi_O=0.4, do_dilution=True,
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    m, opt_p = ref_lam.maximize_growth_rate(params)
    d = ref_lam.results_as_dict(opt_p, params)
    results.append(d)

    # Now without -- seting max_phi_H = 0
    params_nh = GrowthRateOptParams(phi_O=0.4, do_dilution=True, max_phi_H=0,
                                    fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                    fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    m, opt_p = ref_lam.maximize_growth_rate(params_nh)
    d = ref_lam.results_as_dict(opt_p, params_nh)
    results.append(d)

S6_sensitivity_df = pd.DataFrame(results)
S6_sensitivity_df.to_csv('../output/FigX_variable_ZCB.csv')

# Test variable carbon source redox
# TODO: rename ZCorg to ZCred throughout
print('Optimizing over a range of fixed Z_C,red values...')
ZCorgs = np.arange(-3, 3.01, 0.05)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for z in ZCorgs:
    # Test with and without ATP homeostasis -- first with
    ref_lam = lmm.copy()
    ref_lam.set_ZCorg(z)
    # Note we are fixing phi_O here to highlight the contribution of phi_H
    params = GrowthRateOptParams(phi_O=0.4, do_dilution=True,
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    m, opt_p = ref_lam.maximize_growth_rate(params)
    d = ref_lam.results_as_dict(opt_p, params)
    results.append(d)

    # Now without -- seting max_phi_H = 0
    params_nh = GrowthRateOptParams(phi_O=0.4, do_dilution=True, max_phi_H=0,
                                    fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                    fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    m, opt_p = ref_lam.maximize_growth_rate(params_nh)
    d = ref_lam.results_as_dict(opt_p, params_nh)
    results.append(d)

zcorg_sensitivity_df = pd.DataFrame(results)
zcorg_sensitivity_df.to_csv('../output/Fig2B_variable_ZCred.csv')

print('Sweeping pairs of (Z_C,red, S4) values...')
ZCorgs = np.arange(-3, 3.01, 0.05)
S4vals = np.arange(0.2, 1.21, 0.2)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for S4 in S4vals:
    for z in ZCorgs:
        # Test with and without ATP homeostasis -- first with
        ref_lam = lmm.copy()
        ref_lam.set_ZCorg(z)
        ref_lam.set_ATP_yield('reduction', S4)
        # Note we are fixing phi_O here to highlight the contribution of phi_H
        params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True,
                                     fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                     fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
        m, opt_p = ref_lam.maximize_growth_rate(params)
        d = ref_lam.results_as_dict(opt_p, params)
        results.append(d)

        # Now without -- seting max_phi_H = 0
        params_nh = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, max_phi_H=0,
                                        fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                        fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
        m, opt_p = ref_lam.maximize_growth_rate(params_nh)
        d = ref_lam.results_as_dict(opt_p, params_nh)
        results.append(d)

zcorg_sensitivity_var_s4 = pd.DataFrame(results)
zcorg_sensitivity_var_s4.to_csv('../output/Fig4A_variable_ZCorg_var_S4.csv')

print('Sweeping pairs of (Z_C,red, S3) values...')
ZCorgs = np.arange(-3, 3.01, 0.05)
S3vals = np.arange(-0.1, 0.91, 0.2)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for S3 in S3vals:
    for z in ZCorgs:
        # Test with and without ATP homeostasis -- first with
        ref_lam = lmm.copy()
        ref_lam.set_ZCorg(z)
        ref_lam.set_ATP_yield('oxidation', S3)
        # Note we are fixing phi_O here to highlight the contribution of phi_H
        params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True,
                                     fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                     fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
        m, opt_p = ref_lam.maximize_growth_rate(params)
        d = ref_lam.results_as_dict(opt_p, params)
        results.append(d)

        # Now without -- seting max_phi_H = 0
        params_nh = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, max_phi_H=0,
                                        fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                        fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
        m, opt_p = ref_lam.maximize_growth_rate(params_nh)
        d = ref_lam.results_as_dict(opt_p, params_nh)
        results.append(d)

zcorg_sensitivity_var_s3 = pd.DataFrame(results)
zcorg_sensitivity_var_s3.to_csv('../output/Fig4B_variable_ZCorg_var_S3.csv')

print('Sweeping pairs of (Z_C,red, Z_C,B) values...')
ZCorgs = np.arange(-3, 3.01, 0.05)
ZCBs = np.arange(-0.5, 0.51, 0.1)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for zcb in ZCBs:
    for zcorg in ZCorgs:
        # Test with and without ATP homeostasis -- first with
        ref_lam = lmm.copy()
        ref_lam.set_ZCorg(zcorg)
        ref_lam.set_ZCB(zcb)
        # Note we are fixing phi_O here to highlight the contribution of phi_H
        params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True,
                                     fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                     fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
        m, opt_p = ref_lam.maximize_growth_rate(params)
        d = ref_lam.results_as_dict(opt_p, params)
        results.append(d)

        # Now without -- seting max_phi_H = 0
        params_nh = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, max_phi_H=0,
                                        fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                        fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
        m, opt_p = ref_lam.maximize_growth_rate(params_nh)
        d = ref_lam.results_as_dict(opt_p, params_nh)
        results.append(d)

zcorg_sensitivity_var_zcb = pd.DataFrame(results)
zcorg_sensitivity_var_zcb.to_csv('../output/Fig4C_variable_ZCorg_var_ZCB.csv')


print('Loading autotrophy model...')
auto_model_dir = '../models/linear/autotrophy/'
auto_m_fname = path.join(auto_model_dir, 'glucose_auto_molecular_props.csv')
auto_S_fname = path.join(auto_model_dir, 'glucose_auto_stoich_matrix.csv')

# Load models of auto and heterotrophy for comparison
auto_lam = LinearMetabolicModel.FromFiles(auto_m_fname, auto_S_fname)
lam = LinearMetabolicModel.FromFiles(m_fname, S_fname)

print('Comparing autotrophy and heterotrophy over a range of gamma values...')
# Each model is defined by a set of processes that have 
# mass specific catalytic rates (gamma = k/mass). 
# here we set the process masses to a range of values
pmasses = np.logspace(2, 5, 50)  # kDa units

auto_results = []
results = []
for pmass in pmasses:
    # Set the process masses
    auto_lam.set_process_masses(pmass)
    lam.set_process_masses(pmass)
    
    # Make fresh parameters with a new max_lambda_hr
    params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, 
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)

    # Optimizing the heterotrophic growth rate given the parameters
    opt, opt_prob = lam.maximize_growth_rate(params)
    d = lam.results_as_dict(opt_prob, params)
    results.append(d)

    # Optimize the autotrophic growth rate given the parameters
    auto_opt, auto_opt_prob = auto_lam.maximize_growth_rate(params)
    d = auto_lam.results_as_dict(auto_opt_prob, params)
    auto_results.append(d)

auto_gamma_df = pd.DataFrame(auto_results)
auto_gamma_df['mass_kDa'] = pmasses
auto_gamma_df['model'] = 'autotrophy'
gamma_df = pd.DataFrame(results)
gamma_df['mass_kDa'] = pmasses
gamma_df['model'] = 'heterotrophy'
gamma_df = pd.concat([gamma_df, auto_gamma_df])
gamma_df.to_csv('../output/Fig2C_autotrophy_comparison.csv')
