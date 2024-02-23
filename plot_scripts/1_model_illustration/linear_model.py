# -*- coding: utf-8 -*-
# 23.02.2024 Even Moa Myklebust

from utilities import *
import numpy as np
import pandas as pd
import pymc as pm
# Initialize random number generator
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
##############################
# Function argument shapes: 
# X is an (N_patients, P) shaped pandas dataframe
# patient dictionary contains N_patients patients in the same order as X

def linear_model(X, patient_dictionary, name, N_samples=3000, N_tuning=3000, target_accept=0.99, max_treedepth=10, psi_prior="lognormal", FUNNEL_REPARAMETRIZATION=False, method="HMC", scaling=False, t_max_prior_scale=1, likelihood_model="original", model_rho_r_dependancy_on_rho_s=True):
    assert not X.isna().any().any()
    assert not np.isinf(X).any().any()
    assert not X.isna().any().any()
    df = pd.DataFrame(columns=["patient_id", "mprotein_value", "time"])
    for ii in range(len(patient_dictionary)):
        patient = patient_dictionary[ii]
        mprot = patient.Mprotein_values
        times = patient.measurement_times
        for jj in range(len(mprot)):
            single_entry = pd.DataFrame({"patient_id":[ii], "mprotein_value":[mprot[jj]], "time":[times[jj]]})
            df = pd.concat([df, single_entry], ignore_index=True)
    group_id = df["patient_id"].tolist()
    assert not np.isnan(group_id).any()
    assert not np.isinf(group_id).any()
    Y_flat_no_nans = np.array(df["mprotein_value"].tolist())
    assert min(Y_flat_no_nans) >= 0
    assert not np.isnan(Y_flat_no_nans).any()
    assert not np.isinf(Y_flat_no_nans).any()
    t_flat_no_nans = np.array(df["time"].tolist())
    #print(max(t_flat_no_nans)) #1373 with IKEMA
    assert min(t_flat_no_nans) >= 0
    assert not np.isnan(t_flat_no_nans).any()
    assert not np.isinf(t_flat_no_nans).any()
    
    N_patients, P = X.shape
    N_patients = int(N_patients)
    P0 = min(10, max(int(P / 2), 1)) # A guess of the true number of nonzero parameters is needed for defining the global shrinkage parameter
    assert P0 > 0
    assert P > 0
    assert N_patients > 0
    X_not_transformed = X.copy()
    X = X.T
    yi0 = np.zeros(N_patients)
    for ii in range(N_patients):
        yi0[ii] = max(patient_dictionary[ii].Mprotein_values[0], 1e-5)
    assert min(yi0) > 0 #Strictly greater than zero required because we log transform it for the log prior of psi 
    assert not np.isnan(yi0).any()
    assert not np.isinf(yi0).any()

    if psi_prior not in ["lognormal", "normal"]:
        print("Unknown prior option specified for psi; Using 'lognormal' prior")
        psi_prior = "lognormal"

    with pm.Model(coords={"predictors": X_not_transformed.columns.values}) as linear_model:
        # Observation noise (std)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1)

        # alpha
        alpha = pm.Normal("alpha", mu=np.array([np.log(t_max_prior_scale*0.015), np.log(t_max_prior_scale*0.005), np.log(0.01/(1-0.01))]), sigma=1, shape=3)
        # beta (with horseshoe priors):
        # Global shrinkage prior
        tau_rho_r = pm.HalfStudentT("tau_rho_r", nu=2, sigma=P0/(P-P0)*sigma_obs/np.sqrt(N_patients))
        tau_pi_r = pm.HalfStudentT("tau_pi_r", nu=2, sigma=P0/(P-P0)*sigma_obs/np.sqrt(N_patients))
        # Local shrinkage prior
        lam_rho_r = pm.HalfStudentT("lam_rho_r", nu=2, sigma=1, dims="predictors")
        lam_pi_r = pm.HalfStudentT("lam_pi_r", nu=2, sigma=1, dims="predictors")
        c2_rho_r = pm.InverseGamma("c2_rho_r", alpha=1, beta=0.1)
        c2_pi_r = pm.InverseGamma("c2_pi_r", alpha=1, beta=0.1)
        z_rho_r = pm.Normal("z_rho_r", mu=0.0, sigma=1.0, dims="predictors")
        z_pi_r = pm.Normal("z_pi_r", mu=0.0, sigma=1.0, dims="predictors")
        # Shrunken coefficients
        beta_rho_r = pm.Deterministic("beta_rho_r", z_rho_r * tau_rho_r * lam_rho_r * np.sqrt(c2_rho_r / (c2_rho_r + tau_rho_r**2 * lam_rho_r**2)), dims="predictors")
        beta_pi_r = pm.Deterministic("beta_pi_r", z_pi_r * tau_pi_r * lam_pi_r * np.sqrt(c2_pi_r / (c2_pi_r + tau_pi_r**2 * lam_pi_r**2)), dims="predictors")

        # Latent variables theta
        omega = pm.HalfNormal("omega",  sigma=1, shape=3) # Patient variability in theta (std)
        if FUNNEL_REPARAMETRIZATION:
            # Reparametrized to escape/explore the funnel of Hell (https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/):
            #theta_rho_s_offset = pm.Normal('theta_rho_s_offset', mu=0, sigma=1, shape=N_patients)
            #theta_rho_r_offset = pm.Normal('theta_rho_r_offset', mu=0, sigma=1, shape=N_patients)
            #theta_pi_r_offset  = pm.Normal('theta_pi_r_offset',  mu=0, sigma=1, shape=N_patients)
            #theta_rho_s = pm.Deterministic("theta_rho_s", (alpha[0]) + theta_rho_s_offset * omega[0])
            #theta_rho_r = pm.Deterministic("theta_rho_r", (alpha[1] + pm.math.dot(beta_rho_r, X) + pm.math.dot(coef_rho_s, theta_rho_s)) + theta_rho_r_offset * omega[1])
            #theta_pi_r  = pm.Deterministic("theta_pi_r",  (alpha[2] + pm.math.dot(beta_pi_r,  X)) + theta_pi_r_offset  * omega[2])
            pass
        else: 
            # Original
            theta_rho_s = pm.Normal("theta_rho_s", mu= alpha[0], sigma=omega[0], shape=N_patients) # Individual random intercepts in theta to confound effects of X
            if model_rho_r_dependancy_on_rho_s:
                ## Horseshoe-like prior
                #lam_rho_s = pm.HalfStudentT("lam_rho_s", nu=2, sigma=1)
                #z_rho_s = pm.Normal("z_rho_s", mu=0.0, sigma=1.0)
                #coef_rho_s = pm.Deterministic("coef_rho_s", z_rho_s * lam_rho_s)
                # Lasso prior
                lam = 9
                sig = pm.math.sqrt(pm.InverseGamma(name="sig", alpha=3, beta=1))
                coef_rho_s = pm.Laplace("coef_rho_s", mu=0, b=sig/lam)

                theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + pm.math.dot(beta_rho_r, X) + pm.math.dot(coef_rho_s, theta_rho_s), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
            else:
                theta_rho_r = pm.Normal("theta_rho_r", mu= alpha[1] + pm.math.dot(beta_rho_r, X), sigma=omega[1]) # Individual random intercepts in theta to confound effects of X
            theta_pi_r  = pm.Normal("theta_pi_r",  mu= alpha[2] + pm.math.dot(beta_pi_r, X),  sigma=omega[2]) # Individual random intercepts in theta to confound effects of X

        # psi: True M protein at time 0
        # 1) Normal. Fast convergence, but possibly negative tail 
        if psi_prior=="normal":
            psi = pm.Normal("psi", mu=yi0, sigma=sigma_obs, shape=N_patients) # Informative. Centered around the patient specific yi0 with std=observation noise sigma 
        # 2) Lognormal. Works if you give it time to converge
        if psi_prior=="lognormal":
            xi = pm.HalfNormal("xi", sigma=1)
            log_psi = pm.Normal("log_psi", mu=np.log(yi0), sigma=xi, shape=N_patients)
            psi = pm.Deterministic("psi", np.exp(log_psi))

        # Transformed latent variables 
        rho_s = pm.Deterministic("rho_s", -np.exp(theta_rho_s))
        rho_r = pm.Deterministic("rho_r", np.exp(theta_rho_r))
        pi_r  = pm.Deterministic("pi_r", 1/(1+np.exp(-theta_pi_r)))

        if likelihood_model == "Kevins_first":
            print("Likelihood model in linear_model is: Kevin's model no. 1: Logistic growth in resistant cells; limited by total cell count")
            a_param = 1
            K_param_vector = a_param * psi[group_id]
            #b_param_vector = (a_param - pi_r[group_id]) / (1 - pi_r[group_id])
            # b_param_vector = 1 when a = 1
            
            R_zero = psi[group_id] * pi_r[group_id]
            rho_r_grouped = rho_r[group_id]
            psi_grouped = psi[group_id]
            pi_r_grouped = pi_r[group_id]
            rho_s_grouped = rho_s[group_id]

            # nominator
            integral_of_S_from_0_to_t = psi[group_id] * (1 - pi_r[group_id])/rho_s[group_id] * (np.exp(rho_s[group_id] * t_flat_no_nans) - 1)
            #resistant_part_upper = R_zero * np.exp(rho_r[group_id] * (K_param_vector * t_flat_no_nans - b_param_vector * integral_of_S_from_0_to_t))
            resistant_part_upper = R_zero * np.exp(rho_r[group_id] * (K_param_vector * t_flat_no_nans -     1          * integral_of_S_from_0_to_t))
            

            # denominator
            def exp_Kw_minus_rho_r_b_integral_S_dy_dw(w, rho_r_element, psi_element, pi_r_element, rho_s_element, K_param_vector_element, b_param_vector_element):
                return np.exp(K_param_vector_element * w - rho_r_element * b_param_vector_element * psi_element * (1 - pi_r_element)/rho_s_element * (np.exp(rho_s_element * w) - 1))            
            # Attempt to vectorize: 
            #vec_int = np.vectorize(lambda t_index: quad(exp_Kw_minus_rho_r_b_integral_S_dy_dw, 0, t_flat_no_nans[t_index], args=(rho_r_grouped[t_index], psi_grouped[t_index], pi_r_grouped[t_index], rho_s_grouped[t_index], K_param_vector, b_param_vector[t_index])))
            #integral_of_exp_Kw_minus_rho_r_b_integral_S_dy_dw_from_0_to_tt = vec_int([t_index for t_index, _ in enumerate(t_flat_no_nans)])
            # Not working because quad does not support vectorization: 
            #integral_of_exp_Kw_minus_rho_r_b_integral_S_dy_dw_from_0_to_tt = quad(exp_Kw_minus_rho_r_b_integral_S_dy_dw, 0, t_flat_no_nans, args=(rho_r_grouped, psi_grouped, pi_r_grouped, rho_s_grouped, K_param_vector, b_param_vector))
            # non vectorized: 
            # quad does not work 
            #integral_of_exp_Kw_minus_rho_r_b_integral_S_dy_dw_from_0_to_tt = [quad(exp_Kw_minus_rho_r_b_integral_S_dy_dw, 0, t_flat_no_nans[t_index], args=(rho_r_grouped[t_index], psi_grouped[t_index], pi_r_grouped[t_index], rho_s_grouped[t_index], K_param_vector, b_param_vector[t_index])) for t_index, _ in enumerate(t_flat_no_nans)]

            def trapzf(t_index, n=20):
                #x = np.linspace(a, b, n)
                x = np.linspace(0, t_flat_no_nans[t_index], n)
                #y = f(x)
                #y = exp_Kw_minus_rho_r_b_integral_S_dy_dw(x, rho_r_grouped[t_index], psi_grouped[t_index], pi_r_grouped[t_index], rho_s_grouped[t_index], K_param_vector[t_index], b_param_vector[t_index])
                y = exp_Kw_minus_rho_r_b_integral_S_dy_dw(x, rho_r_grouped[t_index], psi_grouped[t_index], pi_r_grouped[t_index], rho_s_grouped[t_index], K_param_vector[t_index], 1)
                #h = (b-a)/(n-1)
                h = t_flat_no_nans[t_index]/(n-1)
                return (h/2)*(y[1:]+y[:-1]).sum()
            
            integral_of_exp_Kw_minus_rho_r_b_integral_S_dy_dw_from_0_to_tt = [trapzf(t_index) for t_index, _ in enumerate(t_flat_no_nans)] # exp_Kw_minus_rho_r_b_integral_S_dy_dw, 0, t_flat_no_nans[t_index], args=(rho_r_grouped[t_index], psi_grouped[t_index], pi_r_grouped[t_index], rho_s_grouped[t_index], K_param_vector, b_param_vector[t_index])) for t_index, _ in enumerate(t_flat_no_nans)]
            resistant_part_lower = 1 + R_zero * rho_r[group_id] * integral_of_exp_Kw_minus_rho_r_b_integral_S_dy_dw_from_0_to_tt

            resistant_part = resistant_part_upper / resistant_part_lower
            sensitive_part = np.exp(rho_s[group_id] * t_flat_no_nans)
            mu_Y = psi[group_id] * (pi_r[group_id]*resistant_part + (1-pi_r[group_id])*sensitive_part)


        elif likelihood_model == "Kevins_second":
            print("Likelihood model in linear_model is: Kevin's model no. 2: Logistic growth in resistant cells; only limited by sensitive cell count")
            ## Kevin's second option (S_zero is psi[group_id])
            #a_param = pm.Gamma("a_param", alpha=2, beta=1)
            #mu_Y = psi[group_id] * (pi_r[group_id] * np.exp(rho_r[group_id]*psi[group_id]*(a_param*t_flat_no_nans - (1 - np.exp(-rho_s[group_id]*t_flat_no_nans))/rho_s[group_id])) + (1-pi_r[group_id])*np.exp(rho_s[group_id]*t_flat_no_nans))

            #a_param = 1
            #S_zero = psi[group_id] * (1-pi_r[group_id])
            #resistant_part = np.exp(rho_r[group_id] * S_zero * (a_param * t_flat_no_nans + (1 - np.exp(rho_s[group_id]*t_flat_no_nans)/(rho_s[group_id]))))
            #sensitive_part = np.exp(rho_s[group_id] * t_flat_no_nans)
            #mu_Y = psi[group_id] * (pi_r[group_id]*resistant_part + (1-pi_r[group_id])*sensitive_part)

            #mu_Y = psi[group_id] * (pi_r[group_id]*np.exp(rho_r[group_id] * psi[group_id] * (1-pi_r[group_id]) * (1 * t_flat_no_nans + (1 - np.exp(rho_s[group_id]*t_flat_no_nans)/(rho_s[group_id])))) + (1-pi_r[group_id])*np.exp(rho_s[group_id] * t_flat_no_nans))
            # With switch to handle rho_s close to zero: 
            # rho_s is a negative number. 
            # First we check that it is not tiny negative: If smaller (more negative) than -1e-10, it's ok. Otherwise set the fraction to t_flat_no_nans (original model)
            fraction_with_rho_s_1 = pm.math.switch(rho_s[group_id] < -1e-10, (1 * t_flat_no_nans + (1 - np.exp(rho_s[group_id]*t_flat_no_nans)/(rho_s[group_id]))), t_flat_no_nans) 
            # Then check that it is not large and negative. If smaller (more negative) than -1e5, set it to -1e5
            fraction_with_rho_s = pm.math.switch(rho_s[group_id] < -1e5, (1 * t_flat_no_nans + (1 - np.exp(-1e5*t_flat_no_nans)/(-1e5))), fraction_with_rho_s_1)
            mu_Y = psi[group_id] * (pi_r[group_id]*np.exp(rho_r[group_id] * psi[group_id] * (1-pi_r[group_id]) * fraction_with_rho_s) + (1-pi_r[group_id])*np.exp(rho_s[group_id] * t_flat_no_nans))

        else:
            print("Likelihood model in linear_model is: Original model without any growth limit or interaction between subpopulations")
            # Original: 
            mu_Y = psi[group_id] * (pi_r[group_id]*np.exp(rho_r[group_id]*t_flat_no_nans) + (1-pi_r[group_id])*np.exp(rho_s[group_id]*t_flat_no_nans))

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu_Y, sigma=sigma_obs, observed=Y_flat_no_nans)
    return linear_model
