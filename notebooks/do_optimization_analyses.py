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
# Here is a model that has no homeostasis (phi_H = 0),
# no maintenance (ATP_maint = 0), but can alter biomass
# composition through phi_O. We set a maximum lambda
# so that we can run the model over a range of lambda values.
lambdas = np.arange(0.1, 4, 0.01)
results = []

for lam_val in lambdas:
    # Make fresh parameters with a new max_lambda_hr
    params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, 
                                 max_lambda_hr=lam_val, max_phi_H=0,
                                 maintenance_cost=0,
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    
    # Optimize the growth rate given the parameters
    opt, opt_prob = lam.maximize_growth_rate(params)
    d = lam.results_as_dict(opt_prob, params)
    results.append(d)

phi_df = pd.DataFrame(results)
phi_df['expected_Jana'] = phi_df['anabolism_gamma']*phi_df['anabolism_phi']
phi_df['expected_lambda'] = MW_C_ATOM*3600*phi_df['expected_Jana']
phi_df.to_csv('../output/Fig2A_variable_lambda.csv', index=False)

### Same as above, but with non-zero maintenance
lambdas = np.arange(0.1, 4, 0.01)
maint = 10 
results = []

for lam_val in lambdas:
    # Make fresh parameters with a new max_lambda_hr
    params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, 
                                 max_lambda_hr=lam_val, max_phi_H=0,
                                 maintenance_cost=maint,
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    
    # Optimize the growth rate given the parameters
    opt, opt_prob = lam.maximize_growth_rate(params)
    d = lam.results_as_dict(opt_prob, params)
    results.append(d)

phi_df = pd.DataFrame(results)
phi_df['expected_Jana'] = phi_df['anabolism_gamma']*phi_df['anabolism_phi']
phi_df['expected_lambda'] = MW_C_ATOM*3600*phi_df['expected_Jana']
phi_df.to_csv('../output/Fig2A_variable_lambda_maint.csv', index=False)


print('Optimizing over a range of fixed phi_red values...')
phi_reds = np.linspace(1e-3, 1e-1, 100)
# Default S4 = 1.0 -- put at midpoint of sweep
S4vals = np.arange(0.5, 1.51, 0.25)
results = []

for phi_r in phi_reds:
    for s4 in S4vals:
        my_lam = lam.copy()
        my_lam.set_ATP_yield('reduction', s4)
        # Make fresh params
        params = GrowthRateOptParams(min_phi_O=0.4, 
                                    do_dilution=True, 
                                    phi_red=phi_r,
                                    fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                    fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
        # Optimize the growth rate given the parameters
        opt, opt_prob = my_lam.maximize_growth_rate(params)
        if opt_prob.status != 'optimal':
            print('Warning: optimization not optimal for phi_red = ', phi_r)
            continue

        d = my_lam.results_as_dict(opt_prob, params)
        results.append(d)

phi_df = pd.DataFrame(results)
phi_df.to_csv('../output/Fig2S1_variable_phi_red.csv', index=False)

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
S6_sensitivity_df.to_csv('../output/FigX_variable_ZCB.csv', index=False)

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
zcorg_sensitivity_df.to_csv('../output/Fig2B_variable_ZCred.csv', index=False)

print('Sweeping pairs of (Z_C,red, S4) values...')
ZCorgs = np.arange(-3, 3.01, 0.05)

# Default S4 = 1.0 -- put at midpoint of sweep
S4vals = np.arange(0.5, 1.51, 0.05)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for S4 in S4vals:
    for z in ZCorgs:
        # Test with and without ATP homeostasis -- first with
        ref_lam = lmm.copy()
        ref_lam.set_ZCorg(z)
        ref_lam.set_ATP_yield('reduction', S4)
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
zcorg_sensitivity_var_s4.to_csv('../output/Fig2_variable_ZCorg_var_S4.csv', index=False)

print('Sweeping pairs of (Z_C,red, S3) values...')
ZCorgs = np.arange(-3, 3.01, 0.05)

# Default S3 = 0.5 ATP/ -- include in sweep
S3vals = np.arange(-0.25, 1.251, 0.25)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)
print("Default S3: ", lmm.get_ATP_yield('oxidation'))

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
zcorg_sensitivity_var_s3.to_csv('../output/Fig3B_variable_ZCorg_var_S3.csv', index=False)

print('Sweeping pairs of (Z_C,red, Z_C,B) values...')
ZCorgs = np.arange(-3, 3.01, 0.05)
# Default ZCB = 0.0 -- put at midpoint of sweep
ZCBs = np.arange(-0.5, 0.51, 0.1)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)
print("Default S4: ", lmm.get_ATP_yield('reduction'))

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
zcorg_sensitivity_var_zcb.to_csv('../output/Fig3C_variable_ZCorg_var_ZCB.csv', index=False)

# Repeat the above, but with a maximum C uptake rate
print('Sweeping pairs of (Z_C,red, Z_C,B) values with a maximum C uptake rate...')

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

# Test maximum C uptake fluxes ranging from 10-90% of the 
# unconstrained maximum from the above sweep
max_C_uptake_fluxes = np.arange(0.1, 0.91, 0.4)*6e-05

for max_C_uptake_flux in max_C_uptake_fluxes:
    for zcb in ZCBs:
        for zcorg in ZCorgs:
            # Test with and without ATP homeostasis -- first with
            ref_lam = lmm.copy()
            ref_lam.set_ZCorg(zcorg)
            ref_lam.set_ZCB(zcb)
            params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True,
                                         max_C_uptake=max_C_uptake_flux,
                                         fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                         fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
            m, opt_p = ref_lam.maximize_growth_rate(params)
            d = ref_lam.results_as_dict(opt_p, params)
            results.append(d)

            # Now seting max_phi_H = 0, i.e. no ATP homeostasis
            params_nh = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True,
                                            max_C_uptake=max_C_uptake_flux, max_phi_H=0,
                                            fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                            fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
            m, opt_p = ref_lam.maximize_growth_rate(params_nh)
            d = ref_lam.results_as_dict(opt_p, params_nh)
            results.append(d)

zcorg_sensitivity_var_zcb = pd.DataFrame(results)
zcorg_sensitivity_var_zcb.to_csv('../output/Fig3C_variable_ZCorg_var_ZCB_max_C_uptake.csv', index=False)

print('Loading autotrophy model...')
auto_model_dir = '../models/linear/autotrophy/'
auto_m_fname = path.join(auto_model_dir, 'glucose_auto_molecular_props.csv')
auto_S_fname = path.join(auto_model_dir, 'glucose_auto_stoich_matrix.csv')

# Load models of auto and heterotrophy for comparison
lam = LinearMetabolicModel.FromFiles(m_fname, S_fname)
auto_lam = LinearMetabolicModel.FromFiles(auto_m_fname, auto_S_fname, heterotroph=False)

# A model of autotrophy where we don't enforce Cred homeostasis at all
auto_lam_ext_C = LinearMetabolicModel.FromFiles(auto_m_fname, auto_S_fname, heterotroph=False)
auto_lam_ext_C.m_df.loc['C_red', 'internal'] = 0

print('Comparing autotrophy and heterotrophy with sampled gamma values...')
# Sample sets of 3 process mass (kDa units) from a lognormal distribution
# for oxidation, reduction, anabolism. This has the effect of changing 
# gamma for these processes, but not for homeostasis/CEF.
np.random.seed(42)
pmasses = np.random.lognormal(mean=np.log(1000), sigma=np.log(3), size=(3,100))
pmasses[np.where(pmasses <= 0)] = 1

auto_results = []
auto_ext_C_results = []
results = []
for idx in range(100):
    # Set the process masses to the sampled values for all models
    for pmass, process in zip(pmasses[:,idx], 'oxidation,reduction,anabolism'.split(',')):
        auto_lam.set_process_mass(process, pmass)
        auto_lam_ext_C.set_process_mass(process, pmass)
        lam.set_process_mass(process, pmass)
    
    # Make fresh parameters with a new max_lambda_hr
    params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, 
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)

    # Optimizing the heterotrophic growth rate given the parameters
    opt, opt_prob = lam.maximize_growth_rate(params)
    d = lam.results_as_dict(opt_prob, params)
    results.append(d)

    # Optimize the autotrophic growth rate given the parameters
    # First run an autotrophic model where we don't enforce mass balance of 
    # internally produced organic carbon. 
    auto_params = params.copy()
    auto_opt, auto_opt_prob = auto_lam_ext_C.maximize_growth_rate(auto_params)
    d = auto_lam_ext_C.results_as_dict(auto_opt_prob, auto_params)
    auto_ext_C_results.append(d)

    # Now models with mass balance of Cred enforced at various concs. 
    # Need to set the reduced C concentration because we dilute 
    # intracellular metabolites and Cred is intracellular in autotrophy.
    # Setting a range of Cred concentrations. Middle value of 1e-6 
    # is a biologically reasonable value of ≈ 1 mM. 
    for c_red in [1e-12, 1e-6, 0.1]:
        auto_params = params.copy()
        auto_params.fixed_C_red = c_red
        auto_opt, auto_opt_prob = auto_lam.maximize_growth_rate(auto_params)
    
        d = auto_lam.results_as_dict(auto_opt_prob, auto_params)
        auto_results.append(d)

# Output files are in the same order -- can match up model runs that way.
auto_gamma_df = pd.DataFrame(auto_results)
auto_gamma_df['model'] = 'autotrophy'
auto_gamma_df.to_csv('../output/Fig2C_autotrophy_samples.csv', index=False)

auto_gamma_ext_C_df = pd.DataFrame(auto_ext_C_results)
auto_gamma_ext_C_df['model'] = 'autotrophy_ext_C'
auto_gamma_ext_C_df.to_csv('../output/Fig2C_autotrophy_ext_C_samples.csv', index=False)

gamma_df = pd.DataFrame(results)
gamma_df['model'] = 'heterotrophy'
gamma_df.to_csv('../output/Fig2C_heterotrophy_samples.csv', index=False)


print('Optimizing the autotrophy model over a range of ZCred values...')
ZCreds = np.arange(-2, 2.01, 0.05)

# Load models of auto and heterotrophy for comparison
auto_lam = LinearMetabolicModel.FromFiles(auto_m_fname, auto_S_fname, heterotroph=False)

results = []
for zcorg in ZCreds:
    my_lam = auto_lam.copy()
    my_lam.set_ZCorg(zcorg)
    # Make fresh parameters with a new max_lambda_hr
    params = GrowthRateOptParams(min_phi_O=0.4, do_dilution=True, 
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)

    # Optimizing the heterotrophic growth rate given the parameters
    opt, opt_prob = my_lam.maximize_growth_rate(params)
    d = my_lam.results_as_dict(opt_prob, params)
    results.append(d)

zcred_auto_df = pd.DataFrame(results)
zcred_auto_df.to_csv('../output/FigSX_autotrophy_ZCred.csv', index=False)


print('Optimizing over a range of min_phi_O values...')
phi_Os = np.arange(0, 0.6, 0.01)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for phi_O in phi_Os:
    # Test with and without ATP homeostasis -- first with
    ref_lam = lmm.copy()
    # Note we are fixing phi_O here to highlight the contribution of phi_H
    params = GrowthRateOptParams(phi_O=phi_O, do_dilution=True,
                                 fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                 fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    m, opt_p = ref_lam.maximize_growth_rate(params)
    d = ref_lam.results_as_dict(opt_p, params)
    results.append(d)

    # Now without -- seting max_phi_H = 0
    params_nh = GrowthRateOptParams(phi_O=phi_O, do_dilution=True, max_phi_H=0,
                                    fixed_ATP=DEFAULT_ATP, fixed_NADH=DEFAULT_NADH,
                                    fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)
    m, opt_p = ref_lam.maximize_growth_rate(params_nh)
    d = ref_lam.results_as_dict(opt_p, params_nh)
    results.append(d)

phi_O_sensitivity_df = pd.DataFrame(results)
phi_O_sensitivity_df.to_csv('../output/Fig2S1_variable_phi_O.csv', index=False)


print('Optimizing over a range of gamma_ana values...')
mKdas = np.logspace(2, 4, 50)

results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)

for ana_kDa in mKdas:
    # Test with and without ATP homeostasis -- first with
    ref_lam = lmm.copy()
    ref_lam.set_process_mass('anabolism', ana_kDa)
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

g_ana_sensitivity_df = pd.DataFrame(results)
g_ana_sensitivity_df.to_csv('../output/Fig2S1_variable_g_ana.csv', index=False)

print('Optimizing over a range of gamma_red values...')
results = []
lmm = LinearMetabolicModel.FromFiles(m_fname, S_fname)
for red_kDa in mKdas:
    # Test with and without ATP homeostasis -- first with
    ref_lam = lmm.copy()
    ref_lam.set_process_mass('reduction', red_kDa)
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

g_red_sensitivity_df = pd.DataFrame(results)
g_red_sensitivity_df.to_csv('../output/Fig2S1_variable_g_red.csv', index=False)
