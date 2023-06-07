SIMULATIONS ONLY

##############################################
## Fig. 1B
##############################################
allmus, allphianas = [], []
# Default params: reliance on respiration.
ZC_org = 2  # NADH yield of carbon source
S4 = 4.0    # ATP yield of respiration
S3 = 0.3    # ATP yield of catabolism
S5 = 1.1    # ATP cost of anabolism
S6 = 0.2    # reduction coefficient of biomass
gamma_cat_inv = 200
gamma_resp_inv = 40
gamma_ana_inv = 200
# everything in units of KM
Corg0 = 1e1
NADH0 = 1e1
O20 = 1e1
ATP0 = 1e1

for indx, t_phi_o in enumerate(np.linspace(0.4, 0.6, 3)):
    mus = []
    phianas = []
    phi_o = t_phi_o
    phi_resp = 0.05

    for t_phi_ana in np.linspace(0.00, 1-phi_o-phi_resp, 200):
        phi_cat = 1-phi_o-phi_resp-t_phi_ana
        phi_ana = t_phi_ana.copy()

        y0 = np.array([ Corg0, NADH0, O20, ATP0 ])
        NUM_METS = len(y0)

        # Evaluation time
        TFINAL = 1e6
        t = np.logspace(-4, 6, 1000 )
        S6 = 1.

        y_sol = solve_ivp(het_model, [ 1e-4, TFINAL ],  y0, t_eval=t, method='Radau' ).y


        nus = np.array([give_nus(y_sol, tpt) for tpt in range(len(t))]).T

        mus.append(nus[-1, -1])
        phianas.append(t_phi_ana)

    mus.append(0.0)
    phianas.append(t_phi_ana)
    allmus.append(mus)
    allphianas.append(phianas)

df_1b = pd.DataFrame({'mus': allmus, 'phianas': allphianas})

##############################################
## Fig. 1C
##############################################
allmus, allnuresps = [], []
tau = 1e-4
# Default params: reliance on respiration.
ZC_org = 2  # NADH yield of carbon source
S4 = 30.0    # ATP yield of respiration
S3 = 0.2    # ATP yield of catabolism
S5 = 1.4   # ATP cost of anabolism
S6 = 0.2    # reduction coefficient of biomass
# everything in units of KM
Corg0 = 1e1
NADH0 = 1e1
O20 = 1e1
ATP0 = 1e1

gamma_cat_inv = 200
gamma_resp_inv = 400
gamma_ana_inv = 200

phi_o = 0.4
phi_cat = 0.3
legend_elements = []

for indx, t_phi_o in enumerate(np.linspace(0.4, 0.6, 3)):
    phi_o = t_phi_o.copy()
    if indx == 1:
        phi_cat = 0.28
    elif indx == 2:
        phi_cat = 0.25
    mus, phianas, phiresps, allatps, nuresps = [], [], [], [], []
    for t_phi_resp in np.linspace(0.00, 1-phi_o-phi_cat, 200):
        phi_resp = t_phi_resp.copy()
        phi_ana = 1-phi_o-phi_cat-phi_resp

        y0 = np.array([ Corg0, NADH0, O20, ATP0 ])
        NUM_METS = len(y0)

        # Evaluation time
        TFINAL = 1e6
        t = np.logspace(-4, 6, 1000 )
        S6 = 0.5

        y_sol = solve_ivp(het_model, [ 1e-4, TFINAL ],  y0, t_eval=t, method='Radau' ).y


        nus = np.array([give_nus(y_sol, tpt) for tpt in range(len(t))]).T

        mus.append(nus[-1, -1])
        phiresps.append(phi_resp)
        nuresps.append((y_sol[1, -1] ** 2) * phi_resp * gamma_resp_inv * tau)
        allatps.append(y_sol[3, -1])

    allnuresps.append(nuresps)
    allmus.append(mus)

df_1c = pd.DataFrame({'mus': allmus, 'nuresps': allnuresps})

##############################################
## Fig. 1D
##############################################
allmus, allphianas = [], []
# gammas are in (g s)/(mol) units.
gamma_cat_inv = 200
gamma_resp_inv = 40
gamma_ana_inv = 200

# Default params: reliance on respiration.
ZC_org = 2  # NADH yield (ZC) of carbon source
S4 = 4.0    # ATP yield of respiration
S3 = 0.3    # ATP yield of catabolism
S5 = 1.1    # ATP cost of anabolism
S6 = 0.2    # reduction coefficient of biomass

tau = 1e-4

cs = ['dodgerblue', 'indianred', 'mediumseagreen']
legend_elements = []
phi_o = 0.6
ZC_org = 2.0

for indx, S3 in enumerate(np.linspace(1.1, 0.6, 2)):
    mus = []
    phianas = []
    phi_resp = 0.12

    for t_phi_ana in np.linspace(0.00, 1-phi_o-phi_resp, 200):
        phi_cat = 1-phi_o-phi_resp-t_phi_ana
        phi_ana = t_phi_ana.copy()

        y0 = np.array([ Corg0, NADH0, O20, ATP0 ])
        NUM_METS = len(y0)

        # Evaluation time
        TFINAL = 1e6
        t = np.logspace(-4, 6, 1000 )
        S6 = 1.

        y_sol = solve_ivp(het_model, [ 1e-4, TFINAL ],  y0, t_eval=t, method='Radau' ).y

        nus = np.array([give_nus(y_sol, tpt) for tpt in range(len(t))]).T

        mus.append(nus[-1, -1])
        phianas.append(t_phi_ana)

    mus.append(0.0)
    phianas.append(t_phi_ana)
    allmus.append(mus)
    allphianas.append(phianas)
  
S3 = 1.1
indx += 1
phi_o = 0.6
ZC_org = 2.05
mus = []
phianas = []
phi_resp = 0.12

for t_phi_ana in np.linspace(0.00, 1-phi_o-phi_resp, 200):
    phi_cat = 1-phi_o-phi_resp-t_phi_ana
    phi_ana = t_phi_ana.copy()

    y0 = np.array([ Corg0, NADH0, O20, ATP0 ])
    NUM_METS = len(y0)

    # Evaluation time
    TFINAL = 1e6
    t = np.logspace(-4, 6, 1000 )
    S6 = 1.

    y_sol = solve_ivp(het_model, [ 1e-4, TFINAL ],  y0, t_eval=t, method='Radau' ).y

    nus = np.array([give_nus(y_sol, tpt) for tpt in range(len(t))]).T

    mus.append(nus[-1, -1])
    phianas.append(t_phi_ana)

mus.append(0.0)
phianas.append(t_phi_ana)
allmus.append(mus)
allphianas.append(phianas)

# resetting carbon source ZC.
ZC_org = 2.0

df_1d = pd.DataFrame({'mus': allmus, 'phianas': allphianas})

##############################################
## Fig. 1E
##############################################
all_mus = []
all_zcs = []

gamma_cat_inv = 200
gamma_resp_inv = 40
gamma_ana_inv = 200


# Default params: reliance on respiration.
NADH0 = 10. # init NADH conc in units of KM
ZC_org = 2  # NADH yield (ZC) of carbon source
S4 = 4.0    # ATP yield of respiration
S3 = 0.3    # ATP yield of catabolism
S5 = 1.1    # ATP cost of anabolism
for indx in range(3):
    mus = []
    zcs = []
    tau = 1e-4
    phi_o = 0.4
    phi_cat = 0.2
    phi_resp = 0.1

    if indx == 1:
        NADH0 = 6.0
    elif indx == 2:
        NADH0 = 14.
        S4 = 20.

    phi_ana = 1 - phi_cat - phi_resp - phi_o

    for S6 in np.linspace(-0.1, 0.5, 100):
        y0 = np.array([Corg0, NADH0, O20, ATP0])
        NUM_METS = len(y0)

        # Evaluation time
        TFINAL = 1e6
        t = np.logspace(-4, 6, 1000)

        y_sol = solve_ivp(het_model, [1e-4, TFINAL], y0, t_eval=t, method='Radau').y

        nus = np.array([give_nus(y_sol, tpt) for tpt in range(len(t))]).T

        mus.append(nus[-1, -1])
        zcs.append(S6)

    all_mus.append(mus)
    all_zcs.append(zcs)

df_1e = pd.DataFrame({'mus': all_mus, 'zcs': all_zcs})
