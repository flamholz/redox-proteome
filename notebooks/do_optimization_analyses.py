import argparse
import numpy as np
import os
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
DEFAULT_C_OX = 1e-5
DEFAULT_RE = 10
DEFAULT_RA = 0.3
DEFAULT_MAINTENANCE = 10
DEFAULT_OPT_VALS = dict(do_dilution=True, dilute_as_sum=True,
                        fixed_ATP=DEFAULT_ATP, fixed_ECH=DEFAULT_NADH,
                        fixed_re=DEFAULT_RE, fixed_ra=DEFAULT_RA)

# Directories
MODEL_BASE_DIR = '../models/linear/'
RESP_MODEL_DIR = path.join(MODEL_BASE_DIR, 'respiration/')
FERM_MODEL_DIR = path.join(MODEL_BASE_DIR, 'fermentation/')
AUTO_MODEL_DIR = path.join(MODEL_BASE_DIR, 'autotrophy/')

# Filenames
resp_m_fname = path.join(RESP_MODEL_DIR, 'glucose_resp_molecular_props.csv')
resp_S_fname = path.join(RESP_MODEL_DIR, 'glucose_resp_stoich_matrix.csv')
ferm_m_fname = path.join(FERM_MODEL_DIR, 'glucose_ferm_molecular_props.csv')
ferm_S_fname = path.join(FERM_MODEL_DIR, 'glucose_ferm_stoich_matrix.csv')
auto_m_fname = path.join(AUTO_MODEL_DIR, 'glucose_auto_molecular_props.csv')
auto_S_fname = path.join(AUTO_MODEL_DIR, 'glucose_auto_stoich_matrix.csv')


def do_main(outdir, overwrite):
    if not path.exists(outdir):
        os.makedirs(outdir)

    print('Loading respiration model...')
    # Load a respiration model -- change no parameters on this one, only on copies
    lam = LinearMetabolicModel.FromFiles(resp_m_fname, resp_S_fname)

    # First analysis -- fix lambda by setting a max.  
    # Model will optimize the biomass composition given this ceiling on lambda.
    print('Optimizing over a range of fixed lambda values...')
    lambdas = np.arange(0.1, 4, 0.01)

    out_fname = path.join(outdir, 'fix_lambda.csv')
    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for lam_val in lambdas:
            # Make fresh parameters with a new max_lambda_hr
            params = GrowthRateOptParams(min_phi_O=0.4, max_lambda_hr=lam_val, max_phi_H=0,
                                         maintenance_cost=0, **DEFAULT_OPT_VALS)
            
            # Optimize the growth rate given the parameters
            opt, opt_prob = lam.maximize_growth_rate(params)
            d = lam.results_as_dict(opt_prob, params)
            results.append(d)

        phi_df = pd.DataFrame(results)
        phi_df['expected_Jana'] = phi_df['anabolism_gamma']*phi_df['anabolism_phi']
        phi_df['expected_lambda'] = MW_C_ATOM*3600*phi_df['expected_Jana']
        phi_df.to_csv(out_fname, index=False)

    # Same as above, but with non-zero maintenance ATP cost
    out_fname = path.join(outdir, 'fix_lambda_maint.csv')

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for lam_val in lambdas:
            params = GrowthRateOptParams(min_phi_O=0.4, max_lambda_hr=lam_val, max_phi_H=0,
                                         maintenance_cost=DEFAULT_MAINTENANCE,
                                         **DEFAULT_OPT_VALS)
            
            # Optimize the growth rate given the parameters
            opt, opt_prob = lam.maximize_growth_rate(params)
            d = lam.results_as_dict(opt_prob, params)
            results.append(d)

        phi_df = pd.DataFrame(results)
        phi_df['expected_Jana'] = phi_df['anabolism_gamma']*phi_df['anabolism_phi']
        phi_df['expected_lambda'] = MW_C_ATOM*3600*phi_df['expected_Jana']
        phi_df.to_csv(out_fname, index=False)

    # Fix phi_red and allow everything else to be optimized
    print('Optimizing over a range of fixed phi_red values...')
    out_fname = path.join(outdir, 'fix_phi_red.csv')
    phi_reds = np.linspace(1e-3, 1e-1, 100)
    # Default S4 = 1.0 -- put at midpoint of sweep
    S4vals = np.arange(0.5, 1.51, 0.25)

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for phi_r in phi_reds:
            for s4 in S4vals:
                my_lam = lam.copy()
                my_lam.set_ATP_yield('reduction', s4)
                params = GrowthRateOptParams(min_phi_O=0.4, phi_red=phi_r, **DEFAULT_OPT_VALS)
                opt, opt_prob = my_lam.maximize_growth_rate(params)
                # TODO: fix by setting the max phi_red to the maximum feasible value
                if opt_prob.status != 'optimal':
                    print('Warning: optimization not optimal for phi_red = ', phi_r)
                    continue
                d = my_lam.results_as_dict(opt_prob, params)
                results.append(d)

        phi_df = pd.DataFrame(results)
        phi_df.to_csv(out_fname, index=False)

    print('Optimizing over a range of fixed Z_C,B values...')
    out_fname = path.join(outdir, 'fix_ZCB.csv')
    # Sweep a range of biomass ZC values
    ZCBs = np.arange(-3, 3.01, 0.05)

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:    
        results = []
        for z in ZCBs:
            # Test with and without ATP homeostasis -- first with
            ref_lam = lam.copy()
            ref_lam.set_ZCB(z)
            # Note we are fixing phi_O here to highlight the contribution of phi_H
            params = GrowthRateOptParams(phi_O=0.4, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params)
            d = ref_lam.results_as_dict(opt_p, params)
            results.append(d)

            # Now without -- seting max_phi_H = 0
            params_nh = GrowthRateOptParams(phi_O=0.4, max_phi_H=0,
                                            **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params_nh)
            d = ref_lam.results_as_dict(opt_p, params_nh)
            results.append(d)

        # Changing the ZCB value affects the ECH stoichiometry of
        # anabolism, S6 = nu^e_ana. Hence the name of the dataframe.
        S6_sensitivity_df = pd.DataFrame(results)
        S6_sensitivity_df.to_csv(out_fname, index=False)

    print('Optimizing over a range of phi_O values...')
    phi_Os = np.arange(0, 0.6, 0.01)
    out_fname = path.join(outdir, 'fix_phi_O.csv')

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for phi_O in phi_Os:
            # Test with and without ATP homeostasis -- first with
            ref_lam = lam.copy()
            # Note we are fixing phi_O here to highlight the contribution of phi_H
            params = GrowthRateOptParams(phi_O=phi_O, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params)
            d = ref_lam.results_as_dict(opt_p, params)
            results.append(d)

            # Now without -- seting max_phi_H = 0
            params_nh = GrowthRateOptParams(phi_O=phi_O, max_phi_H=0, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params_nh)
            d = ref_lam.results_as_dict(opt_p, params_nh)
            results.append(d)

        phi_O_sensitivity_df = pd.DataFrame(results)
        phi_O_sensitivity_df.to_csv(out_fname, index=False)

    print('Optimizing over a range of gamma_ana values...')
    # Sweep a range of anabolism gamma values by setting a 
    # fixed anabolism process mass in kDa units.
    mKdas = np.logspace(2, 4, 50)

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for ana_kDa in mKdas:
            # Test with and without ATP homeostasis -- first with
            ref_lam = lam.copy()
            ref_lam.set_process_mass('anabolism', ana_kDa)
            # Note we are fixing phi_O here to highlight the contribution of phi_H
            params = GrowthRateOptParams(phi_O=0.4, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params)
            d = ref_lam.results_as_dict(opt_p, params)
            results.append(d)

            # Now without -- seting max_phi_H = 0
            params_nh = GrowthRateOptParams(phi_O=0.4, max_phi_H=0, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params_nh)
            d = ref_lam.results_as_dict(opt_p, params_nh)
            results.append(d)

        g_ana_sensitivity_df = pd.DataFrame(results)
        g_ana_sensitivity_df.to_csv(out_fname, index=False)

    print('Optimizing over a range of gamma_red values...')
    out_fname = path.join(outdir, 'fix_g_red.csv')
    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for red_kDa in mKdas:
            # Test with and without ATP homeostasis -- first with
            ref_lam = lam.copy()
            ref_lam.set_process_mass('reduction', red_kDa)
            # Note we are fixing phi_O here to highlight the contribution of phi_H
            params = GrowthRateOptParams(phi_O=0.4, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params)
            d = ref_lam.results_as_dict(opt_p, params)
            results.append(d)

            # Now without -- seting max_phi_H = 0
            params_nh = GrowthRateOptParams(phi_O=0.4, max_phi_H=0, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params_nh)
            d = ref_lam.results_as_dict(opt_p, params_nh)
            results.append(d)

        g_red_sensitivity_df = pd.DataFrame(results)
        g_red_sensitivity_df.to_csv(out_fname, index=False)

    # Test variable carbon source redox
    print('Optimizing over a range of fixed Z_C,red values...')
    ZCreds = np.arange(-3, 3.01, 0.05)
    out_fname = path.join(outdir, 'fix_ZCred.csv')

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for z in ZCreds:
            # Test with and without ATP homeostasis -- first with
            ref_lam = lam.copy()
            ref_lam.set_ZCred(z)
            # Note we are fixing phi_O here to highlight the contribution of phi_H
            # TODO: what happens when phi_O is not fixed??????
            # Analytics say it only affect maintenance term. 
            params = GrowthRateOptParams(phi_O=0.4, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params)
            d = ref_lam.results_as_dict(opt_p, params)
            results.append(d)

            # Now without -- seting max_phi_H = 0
            params_nh = GrowthRateOptParams(phi_O=0.4, max_phi_H=0, **DEFAULT_OPT_VALS)
            m, opt_p = ref_lam.maximize_growth_rate(params_nh)
            d = ref_lam.results_as_dict(opt_p, params_nh)
            results.append(d)

        zcred_sensitivity_df = pd.DataFrame(results)
        zcred_sensitivity_df.to_csv(out_fname, index=False)

    # Default S5 = nu^a_ana = 0.3 ATP/C in anabolism 
    # ~5 ATP/amino acid from the translation machinery gives ≈1 ATP/C (≈5 C/AA).
    # So 0.3 is a lower bound, realistically. Cossetto et al. bioRxiv 2024 
    # gives 10 ATP/C from meta-analysis of culture data. 
    print('Sweeping pairs of (Z_C,red, S5) values...')
    out_fname = path.join(outdir, 'fix_ZCred_S5.csv')
    S5vals = np.concatenate([np.arange(0.1, 2.0, 0.1),
                             np.arange(2, 10.1, 0.5)])

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for S5 in S5vals:
            for z in ZCreds:
                # Test with and without ATP homeostasis -- first with
                ref_lam = lam.copy()
                ref_lam.set_ZCred(z)
                # we consider negative S5 values as anabolism consumes energy
                ref_lam.set_ATP_yield('anabolism', -S5)
                params = GrowthRateOptParams(min_phi_O=0.4, **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params)
                d = ref_lam.results_as_dict(opt_p, params)
                results.append(d)

                # Now without -- seting max_phi_H = 0
                params_nh = GrowthRateOptParams(min_phi_O=0.4, max_phi_H=0, **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params_nh)
                d = ref_lam.results_as_dict(opt_p, params_nh)
                results.append(d)

        zcred_sensitivity_var_s5 = pd.DataFrame(results)
        zcred_sensitivity_var_s5.to_csv(out_fname, index=False)

    print('Sweeping pairs of (Z_C,red, S4) values...')
    # Default S4 = nu^a_red = 1.0 ATP/electron -- put at midpoint of sweep
    S4vals = np.arange(0.1, 2.01, 0.05)
    out_fname = path.join(outdir, 'fix_ZCred_S4.csv')

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []

        for S4 in S4vals:
            for z in ZCreds:
                # Test with and without ATP homeostasis -- first with
                ref_lam = lam.copy()
                ref_lam.set_ZCred(z)
                ref_lam.set_ATP_yield('reduction', S4)
                params = GrowthRateOptParams(min_phi_O=0.4, **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params)
                d = ref_lam.results_as_dict(opt_p, params)
                results.append(d)

                # Now without -- seting max_phi_H = 0
                params_nh = GrowthRateOptParams(min_phi_O=0.4, max_phi_H=0, **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params_nh)
                d = ref_lam.results_as_dict(opt_p, params_nh)
                results.append(d)

        zcred_sensitivity_var_s4 = pd.DataFrame(results)
        zcred_sensitivity_var_s4.to_csv(out_fname, index=False)

    print('Sweeping pairs of (Z_C,red, S3) values...')
    # Default S3 = nu^a_ox 0.5 ATP/C -- include in sweep
    # Notice the use of negative S3 values. In principle, ATP can be
    # consumed in oxidation, so long as it is recouped in reduction.
    S3vals = np.arange(-0.25, 1.251, 0.1)
    out_fname = path.join(outdir, 'fix_ZCred_S3.csv')

    if not path.exists(out_fname) or overwrite:
        results = []
        for S3 in S3vals:
            for z in ZCreds:
                # Test with and without ATP homeostasis -- first with
                ref_lam = lam.copy()
                ref_lam.set_ZCred(z)
                ref_lam.set_ATP_yield('oxidation', S3)
                # Note we are fixing phi_O here to highlight the contribution of phi_H
                params = GrowthRateOptParams(min_phi_O=0.4, **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params)
                d = ref_lam.results_as_dict(opt_p, params)
                results.append(d)

                # Now without -- seting max_phi_H = 0
                params_nh = GrowthRateOptParams(min_phi_O=0.4, max_phi_H=0,
                                                **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params_nh)
                d = ref_lam.results_as_dict(opt_p, params_nh)
                results.append(d)

        zcred_sensitivity_var_s3 = pd.DataFrame(results)
        zcred_sensitivity_var_s3.to_csv(out_fname, index=False)

    print('Sweeping pairs of (Z_C,red, Z_C,B) values...')
    ZCreds = np.arange(-3, 3.01, 0.05)
    # Default ZCB = 0.0 -- put at midpoint of sweep
    ZCBs = np.arange(-0.5, 0.51, 0.1)
    out_fname = path.join(outdir, 'fix_ZCred_ZCB.csv')

    if not path.exists(out_fname) or overwrite:
        results = []

        for zcb in ZCBs:
            for zcred in ZCreds:
                # Test with and without ATP homeostasis -- first with
                ref_lam = lam.copy()
                ref_lam.set_ZCred(zcred)
                ref_lam.set_ZCB(zcb)
                # Note we are fixing phi_O here to highlight the contribution of phi_H
                params = GrowthRateOptParams(min_phi_O=0.4, **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params)
                d = ref_lam.results_as_dict(opt_p, params)
                results.append(d)

                # Now without -- seting max_phi_H = 0
                params_nh = GrowthRateOptParams(min_phi_O=0.4, max_phi_H=0, **DEFAULT_OPT_VALS)
                m, opt_p = ref_lam.maximize_growth_rate(params_nh)
                d = ref_lam.results_as_dict(opt_p, params_nh)
                results.append(d)

        zcred_sensitivity_var_zcb = pd.DataFrame(results)
        zcred_sensitivity_var_zcb.to_csv(out_fname, index=False)

    # Repeat the above, but setting a maximum C uptake rate
    print('Sweeping pairs of (Z_C,red, Z_C,B) values with a maximum C uptake rate...')
    out_fname = path.join(outdir, 'fix_ZCred_ZCB_max_C_uptake.csv')

    if not path.exists(out_fname) or overwrite:
        results = []
        # Test maximum C uptake fluxes ranging from 10-90% of the 
        # unconstrained maximum from the above sweep
        max_C_uptake_fluxes = np.arange(0.1, 0.91, 0.4)*6e-05

        for max_C_uptake_flux in max_C_uptake_fluxes:
            for zcb in ZCBs:
                for zcred in ZCreds:
                    # Test with and without ATP homeostasis -- first with
                    ref_lam = lam.copy()
                    ref_lam.set_ZCred(zcred)
                    ref_lam.set_ZCB(zcb)
                    params = GrowthRateOptParams(min_phi_O=0.4, max_C_uptake=max_C_uptake_flux,
                                                **DEFAULT_OPT_VALS)
                    m, opt_p = ref_lam.maximize_growth_rate(params)
                    d = ref_lam.results_as_dict(opt_p, params)
                    results.append(d)

                    # Now seting max_phi_H = 0, i.e. no ATP homeostasis
                    params_nh = GrowthRateOptParams(min_phi_O=0.4, max_C_uptake=max_C_uptake_flux, max_phi_H=0,
                                                    **DEFAULT_OPT_VALS)
                    m, opt_p = ref_lam.maximize_growth_rate(params_nh)
                    d = ref_lam.results_as_dict(opt_p, params_nh)
                    results.append(d)

        zcred_sensitivity_var_zcb = pd.DataFrame(results)
        zcred_sensitivity_var_zcb.to_csv(out_fname, index=False)

    print('Loading autotrophy model...')
    # Load models of auto and heterotrophy for comparison
    resp_lam = LinearMetabolicModel.FromFiles(resp_m_fname, resp_S_fname)
    auto_lam = LinearMetabolicModel.FromFiles(auto_m_fname, auto_S_fname, heterotroph=False)

    # A model of autotrophy where we don't enforce Cred homeostasis at all
    auto_lam_ext_C = auto_lam.copy()
    auto_lam_ext_C.m_df.loc['C_red', 'internal'] = 0

    print('Comparing autotrophy and respiration with sampled gamma values...')
    # Sample sets of 3 process mass (kDa units) from a lognormal distribution
    # for oxidation, reduction, anabolism. This has the effect of changing 
    # gamma for these processes, but not for homeostasis/AEF.
    np.random.seed(42)
    pmasses = np.random.lognormal(mean=np.log(1000), sigma=np.log(3), size=(3,100))
    pmasses[np.where(pmasses <= 0)] = 1

    # Parameters for optimization
    params = GrowthRateOptParams(min_phi_O=0.4, **DEFAULT_OPT_VALS)
    Cred_concs = [1e-12, 1e-6, 0.1]

    # Output filenames
    out_fname_auto = path.join(outdir, 'auto_sampling.csv')
    out_fname_auto_ext_C = path.join(outdir, 'auto_sampling_ext_C.csv')
    out_fname_resp = path.join(outdir, 'respiration_sampling_auto_comparison.csv')

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname_auto) or overwrite:
        auto_results = []
        auto_ext_C_results = []
        results = []
        for idx in range(100):
            # Set the process masses to the sampled values for all models
            for pmass, process in zip(pmasses[:,idx], 'oxidation,reduction,anabolism'.split(',')):
                auto_lam.set_process_mass(process, pmass)
                auto_lam_ext_C.set_process_mass(process, pmass)
                resp_lam.set_process_mass(process, pmass)
            
            # Optimizing the heterotrophic growth rate given the parameters
            opt, opt_prob = resp_lam.maximize_growth_rate(params)
            d = resp_lam.results_as_dict(opt_prob, params)
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
            for c_red in Cred_concs:
                auto_params = params.copy()
                auto_params.fixed_C_red = c_red
                auto_opt, auto_opt_prob = auto_lam.maximize_growth_rate(auto_params)
            
                d = auto_lam.results_as_dict(auto_opt_prob, auto_params)
                auto_results.append(d)

        # Output files are in the same order -- can match up model runs that way.
        auto_gamma_df = pd.DataFrame(auto_results)
        auto_gamma_df['model'] = 'autotrophy'
        auto_gamma_df.to_csv(out_fname_auto, index=False)

        # Why did I call this "ext"?
        auto_gamma_ext_C_df = pd.DataFrame(auto_ext_C_results)
        auto_gamma_ext_C_df['model'] = 'autotrophy_ext_C'
        auto_gamma_ext_C_df.to_csv(out_fname_auto_ext_C, index=False)

        gamma_df = pd.DataFrame(results)
        gamma_df['model'] = 'respiration'
        gamma_df.to_csv(out_fname_resp, index=False)

    print('Optimizing the autotrophy model over a range of ZCred values...')
    ZCreds = np.arange(-2, 2.01, 0.05)
    out_fname = path.join(outdir, 'autotrophy_fix_ZCred.csv')

    # Load models of auto and heterotrophy for comparison
    auto_lam = LinearMetabolicModel.FromFiles(auto_m_fname, auto_S_fname, heterotroph=False)

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname) or overwrite:
        results = []
        for zcred in ZCreds:
            my_lam = auto_lam.copy()
            my_lam.set_ZCred(zcred)
            # Make fresh parameters with a new max_lambda_hr
            params = GrowthRateOptParams(min_phi_O=0.4, **DEFAULT_OPT_VALS)

            # Optimizing the heterotrophic growth rate given the parameters
            opt, opt_prob = my_lam.maximize_growth_rate(params)
            d = my_lam.results_as_dict(opt_prob, params)
            results.append(d)

        zcred_auto_df = pd.DataFrame(results)
        zcred_auto_df.to_csv(out_fname, index=False)
    
    print('Loading fermentation model...')
    # Load models of auto and heterotrophy for comparison
    resp_lam = LinearMetabolicModel.FromFiles(resp_m_fname, resp_S_fname)
    ferm_lam = LinearMetabolicModel.FromFiles(ferm_m_fname, ferm_S_fname)

    # A model of fermentation where we don't enforce Cox homeostasis at all
    ferm_lam_ext_C = LinearMetabolicModel.FromFiles(ferm_m_fname, ferm_S_fname, heterotroph=False)
    ferm_lam_ext_C.m_df.loc['C_ox', 'internal'] = 0

    print('Comparing fermentation and respiration with sampled gamma values...')
    # Use sampled process masses from above from the comparison of autotrophy and respiration

    # Parameters for the optimization of resp and ferm models
    params = GrowthRateOptParams(min_phi_O=0.4, **DEFAULT_OPT_VALS)
    Cox_concs = np.logspace(-1, 1, 3)*DEFAULT_C_OX
    print('Cox concentrations: ', Cox_concs)

    # Output filenames
    out_fname_ferm = path.join(outdir, 'ferm_sampling.csv')
    out_fname_ferm_ext_C = path.join(outdir, 'ferm_sampling_ext_C.csv')
    out_fname_resp = path.join(outdir, 'respiration_sampling_ferm_comparison.csv')

    # Skip this optimization if output exists and we're not overwriting
    if not path.exists(out_fname_ferm) or overwrite:
        ferm_results = []
        ferm_ext_C_results = []
        results = []
        for idx in range(100):
            # Set the process masses to the sampled values for all models
            for pmass, process in zip(pmasses[:,idx], 'oxidation,reduction,anabolism'.split(',')):
                ferm_lam.set_process_mass(process, pmass)
                ferm_lam_ext_C.set_process_mass(process, pmass)
                resp_lam.set_process_mass(process, pmass)
            
            # Optimizing the heterotrophic growth rate given the parameters
            _, opt_prob = resp_lam.maximize_growth_rate(params)
            d = resp_lam.results_as_dict(opt_prob, params)
            results.append(d)

            # Optimize the fermentative growth rate given the parameters
            # First run a fermentative model where we don't enforce mass balance of 
            # internally produced Cox. 
            ferm_params = params.copy()
            _, ferm_opt_prob = ferm_lam_ext_C.maximize_growth_rate(ferm_params)
            d = ferm_lam_ext_C.results_as_dict(ferm_opt_prob, ferm_params)
            ferm_ext_C_results.append(d)

            # Now models with mass balance of Cox enforced at various concs. 
            for c_ox in Cox_concs:
                ferm_params.fixed_C_ox = c_ox
                _, ferm_opt_prob = ferm_lam.maximize_growth_rate(ferm_params)
                d = ferm_lam.results_as_dict(ferm_opt_prob, ferm_params)
                ferm_results.append(d)

        # Output files are in the same order -- can match up model runs that way.
        ferm_gamma_df = pd.DataFrame(ferm_results)
        ferm_gamma_df['model'] = 'fermentation'
        ferm_gamma_df.to_csv(out_fname_ferm, index=False)

        # Why did I call this "ext"?
        ferm_gamma_ext_C_df = pd.DataFrame(ferm_ext_C_results)
        ferm_gamma_ext_C_df['model'] = 'fermentation_ext_C'
        ferm_gamma_ext_C_df.to_csv(out_fname_ferm_ext_C, index=False)

        gamma_df = pd.DataFrame(results)
        gamma_df['model'] = 'respiration'
        gamma_df.to_csv(out_fname_resp, index=False)


def main():
    parser = argparse.ArgumentParser(description='Run optimization analyses.')
    parser.add_argument('--outdir', type=str, default='../output/linear_optimization/',
                        help='Directory to save output files.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files.')
    args = parser.parse_args()

    do_main(args.outdir, args.overwrite)

if __name__ == '__main__':
    main()