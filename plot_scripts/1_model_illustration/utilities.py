# -*- coding: utf-8 -*-
# 23.02.2024 Even Moa Myklebust

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import arviz as az

class Treatment:
    def __init__(self, start, end, id):
        self.start = start
        self.end = end
        self.id = id

class Parameters: 
    def __init__(self, Y_0, pi_r, g_r, g_s, k_1, sigma):
        self.Y_0 = Y_0 # M protein value at start of treatment
        self.pi_r = pi_r # Fraction of resistant cells at start of treatment 
        self.g_r = g_r # Growth rate of resistant cells
        self.g_s = g_s # Growth rate of sensitive cells in absence of treatment
        self.k_1 = k_1 # Additive effect of treatment on growth rate of sensitive cells
        self.sigma = sigma # Standard deviation of measurement noise
    def to_array_without_sigma(self):
        return np.array([self.Y_0, self.pi_r, self.g_r, self.g_s, self.k_1])
    def to_array_with_sigma(self):
        return np.array([self.Y_0, self.pi_r, self.g_r, self.g_s, self.k_1, self.sigma])
    def to_array_for_prediction(self):
        return np.array([self.pi_r, self.g_r, (self.g_s - self.k_1)])
    def to_prediction_array_composite_g_s_and_K_1(self):
        return [self.pi_r, self.g_r, (self.g_s - self.k_1)]

class Patient: 
    def __init__(self, parameters, measurement_times, treatment_history, covariates = [], name = "nn"):
        self.measurement_times = measurement_times
        self.treatment_history = treatment_history
        self.Mprotein_values = measure_Mprotein_with_noise(parameters, self.measurement_times, self.treatment_history)
        self.covariates = covariates
        self.name = name
        self.longitudinal_data = {} # dictionary where the keys are covariate_names, the entries numpy arrays! 
        self.longitudinal_times = {} # matching time arrays for each covariate
        self.pfs = []
        self.mrd_results = []
        self.mrd_times = []
    def get_measurement_times(self):
        return self.measurement_times
    def get_treatment_history(self):
        return self.treatment_history
    def get_Mprotein_values(self):
        return self.Mprotein_values
    def get_covariates(self):
        return self.covariates


#####################################
# Generative models for simulated data
#####################################
# Efficient implementation 
# Simulates M protein value at times [t + delta_T]_i
# Y_t is the M protein level at start of time interval
def generative_model(Y_t, params, delta_T_values, drug_effect):
    return Y_t * params.pi_r * np.exp(params.g_r * delta_T_values) + Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * delta_T_values)

def generate_resistant_Mprotein(Y_t, params, delta_T_values, drug_effect):
    return Y_t * params.pi_r * np.exp(params.g_r * delta_T_values)

def generate_sensitive_Mprotein(Y_t, params, delta_T_values, drug_effect):
    return Y_t * (1-params.pi_r) * np.exp((params.g_s - drug_effect) * delta_T_values)

def get_pi_r_after_time_has_passed(params, measurement_times, treatment_history):
    Mprotein_values = np.zeros_like(measurement_times, dtype=float) #without dtype you get ints
    # Adding a small epsilon to Y and pi_r to improve numerical stability
    epsilon_value = 1e-15
    Y_t = params.Y_0# + epsilon_value
    pi_r_t = params.pi_r# + epsilon_value
    t_params = Parameters(Y_t, pi_r_t, params.g_r, params.g_s, params.k_1, params.sigma)
    for treat_index in range(len(treatment_history)):
        # Find the correct drug effect k_1
        this_treatment = treatment_history[treat_index]
        if this_treatment.id == 0:
            drug_effect = 0
        #elif this_treatment.id == 1:
        # With inference only for individual combinations at a time, it is either 0 or "treatment on", which is k1
        else:
            drug_effect = t_params.k_1
        #else:
        #    sys.exit("Encountered treatment id with unspecified drug effect in function: measure_Mprotein")
        
        # Filter that selects measurement times occuring while on this treatment line
        correct_times = (measurement_times >= this_treatment.start) & (measurement_times <= this_treatment.end)
        
        delta_T_values = measurement_times[correct_times] - this_treatment.start
        # Add delta T for (end - start) to keep track of Mprotein at end of treatment
        delta_T_values = np.concatenate((delta_T_values, np.array([this_treatment.end - this_treatment.start])))

        # Calculate Mprotein values
        # resistant 
        resistant_mprotein = generate_resistant_Mprotein(Y_t, t_params, delta_T_values, drug_effect)
        # sensitive
        sensitive_mprotein = generate_sensitive_Mprotein(Y_t, t_params, delta_T_values, drug_effect)
        # summed
        recorded_and_endtime_mprotein_values = resistant_mprotein + sensitive_mprotein
        # Assign M protein values for measurement times that are in this treatment period
        Mprotein_values[correct_times] = recorded_and_endtime_mprotein_values[0:-1]
        # Store Mprotein value at the end of this treatment:
        Y_t = recorded_and_endtime_mprotein_values[-1]# + epsilon_value
        pi_r_t = resistant_mprotein[-1] / (resistant_mprotein[-1] + sensitive_mprotein[-1] + epsilon_value) # Add a small number to keep numerics ok
        t_params = Parameters(Y_t, pi_r_t, t_params.g_r, t_params.g_s, t_params.k_1, t_params.sigma)
    return Mprotein_values, pi_r_t

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_noiseless(params, measurement_times, treatment_history):
    Mprotein_values, pi_r_after_time_has_passed = get_pi_r_after_time_has_passed(params, measurement_times, treatment_history)
    return Mprotein_values

# Input: a Parameter object, a numpy array of time points in days, a list of back-to-back Treatment objects
def measure_Mprotein_with_noise(params, measurement_times, treatment_history):
    # Return true M protein value + Noise
    noise_array = np.random.normal(0, params.sigma, len(measurement_times))
    noisy_observations = measure_Mprotein_noiseless(params, measurement_times, treatment_history) + noise_array
    # thresholded at 0
    return np.array([max(0, value) for value in noisy_observations])


################################
# Data simulation
################################

def generate_simulated_patients(measurement_times, treatment_history, true_sigma_obs, N_patients_local, P, get_expected_theta_from_X, true_omega, true_omega_for_psi, seed, RANDOM_EFFECTS, DIFFERENT_LENGTHS=True, USUBJID=False, simulate_rho_r_dependancy_on_rho_s=False, coef_rho_s_rho_r=0, psi_population=50, crop_after_pfs=False):
    np.random.seed(seed)
    #X_mean = np.repeat(0,P)
    #X_std = np.repeat(0.5,P)
    #X = np.random.normal(X_mean, X_std, size=(N_patients_local,P))
    X = np.random.uniform(-1, 1, size=(N_patients_local,P))
    X = pd.DataFrame(X, columns = ["Covariate "+str(ii+1) for ii in range(P)])
    expected_theta_1, expected_theta_2, expected_theta_3 = get_expected_theta_from_X(X, simulate_rho_r_dependancy_on_rho_s, coef_rho_s_rho_r)
    if USUBJID:
        X["USUBJID"] = ["Patient " + x for x in X.index.map(str)]

    # Set the seed again to make the random effects not change with P
    np.random.seed(seed+1)
    if RANDOM_EFFECTS:
        true_theta_rho_s = np.random.normal(expected_theta_1, true_omega[0])
        true_theta_rho_r = np.random.normal(expected_theta_2, true_omega[1])
        true_theta_pi_r  = np.random.normal(expected_theta_3, true_omega[2])
    else:
        true_theta_rho_s = expected_theta_1
        true_theta_rho_r = expected_theta_2
        true_theta_pi_r  = expected_theta_3

    # Set the seed again to get identical observation noise irrespective of random effects or not
    np.random.seed(seed+2)
    true_theta_psi = np.random.normal(np.log(psi_population), true_omega_for_psi, size=N_patients_local)
    true_rho_s = - np.exp(true_theta_rho_s)
    true_rho_r = np.exp(true_theta_rho_r)
    true_pi_r  = 1/(1+np.exp(-true_theta_pi_r))
    true_psi = np.exp(true_theta_psi)

    # Set seed again to give patient random Numbers of M protein
    np.random.seed(seed+3)
    parameter_dictionary = {}
    patient_dictionary = {}
    for training_instance_id in range(N_patients_local):
        psi_patient_i   = true_psi[training_instance_id]
        pi_r_patient_i  = true_pi_r[training_instance_id]
        rho_r_patient_i = true_rho_r[training_instance_id]
        rho_s_patient_i = true_rho_s[training_instance_id]
        these_parameters = Parameters(Y_0=psi_patient_i, pi_r=pi_r_patient_i, g_r=rho_r_patient_i, g_s=rho_s_patient_i, k_1=0, sigma=true_sigma_obs)
        # Remove some measurement times from the end: 
        if DIFFERENT_LENGTHS:
            M_ii = np.random.randint(min(3,len(measurement_times)), len(measurement_times)+1)
        else:
            M_ii = len(measurement_times)
        measurement_times_ii = measurement_times[:M_ii]
        this_patient = Patient(these_parameters, measurement_times_ii, treatment_history, name=str(training_instance_id))
        patient_dictionary["Patient " + str(training_instance_id)] = this_patient
        # From server: 
        #patient_dictionary[training_instance_id] = this_patient
        parameter_dictionary[training_instance_id] = these_parameters
        #plot_true_mprotein_with_observations_and_treatments_and_estimate(these_parameters, this_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=str(training_instance_id), savename="./plots/Bayes_simulated_data/"+str(training_instance_id)
    if crop_after_pfs:
        true_pfs_complete_patient_dictionary = get_true_pfs_new(patient_dictionary, time_scale=1, M_scale=1)
        # Crop measurements afer progression: 
        for ii, patient in enumerate(patient_dictionary.values()):
            if true_pfs_complete_patient_dictionary[ii] > 0:
                patient.Mprotein_values = patient.Mprotein_values[patient.measurement_times <= true_pfs_complete_patient_dictionary[ii]] # + 2*28]
                patient.measurement_times = patient.measurement_times[patient.measurement_times <= true_pfs_complete_patient_dictionary[ii]] # + 2*28]
    return X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s
    #return X, patient_dictionary, parameter_dictionary, expected_theta_1, true_theta_rho_s, true_rho_s, expected_theta_2, true_theta_rho_r, true_rho_r, expected_theta_3, true_theta_pi_r, true_pi_r

def get_expected_theta_from_X_2_0(X, simulate_rho_r_dependancy_on_rho_s=False, coef_rho_s_rho_r=0): # pi and rho_s
    # No interactions between covariates. Because we only compare the covariate-less with the linear model, not the BNN or lin+interactions model
    # These parameters have covariates with an effect: 
    # rho_s: No. Because not important to predict. We only predict relapse. 
    # rho_r: Yes. From covariates 0, 1 and 2
    # pi_r: Yes. From covariates 0, 1 and 2

    N_patients_local, P = X.shape
    # These are the true parameters for a patient with all covariates equal to 0:
    rho_s_population = -0.015
    rho_r_population = 0.005
    pi_r_population = 0.01
    theta_rho_s_population_for_x_equal_to_zero = np.log(-rho_s_population)
    theta_rho_r_population_for_x_equal_to_zero = np.log(rho_r_population)
    theta_pi_r_population_for_x_equal_to_zero  = np.log(pi_r_population/(1-pi_r_population))

    true_alpha = np.array([theta_rho_s_population_for_x_equal_to_zero, theta_rho_r_population_for_x_equal_to_zero, theta_pi_r_population_for_x_equal_to_zero])
    #true_beta_rho_s = np.zeros(P)
    #true_beta_rho_s[0] = 0
    #true_beta_rho_s[1] = 0
    #true_beta_rho_s[2] = 0

    true_beta_rho_r = np.zeros(P)
    true_beta_rho_r[0] = 0.7
    true_beta_rho_r[1] = 0.6
    true_beta_rho_r[2] = 0.2

    true_beta_pi_r = np.zeros(P)
    true_beta_pi_r[0] = 0.3
    true_beta_pi_r[1] = 0.4
    true_beta_pi_r[2] = 0.5

    expected_theta_1 = np.reshape(true_alpha[0] + np.dot(X, np.zeros(P)), (N_patients_local,1))
    expected_theta_2 = np.reshape(true_alpha[1] + np.dot(X, true_beta_rho_r), (N_patients_local,1))
    if simulate_rho_r_dependancy_on_rho_s:
        expected_theta_2 = expected_theta_2 + coef_rho_s_rho_r*expected_theta_1
    expected_theta_3 = np.reshape(true_alpha[2] + np.dot(X, true_beta_pi_r), (N_patients_local,1))
    return expected_theta_1, expected_theta_2, expected_theta_3


#####################################
# Posterior evaluation
#####################################
# Convergence checks
def quasi_geweke_test(idata, model_name, first=0.1, last=0.5, intervals=20):
    if first+last > 1:
        print("Overriding input since first+last>1. New first, last = 0.1, 0.5")
        first, last = 0.1, 0.5
    print("Running Geweke test...")
    convergence_flag = True
    if model_name == "linear":
        # 'beta_rho_s'
        var_names = ['alpha', 'beta_rho_r', 'beta_pi_r', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    elif model_name == "BNN":
        var_names = ['alpha', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    else:
        var_names = ['alpha', 'omega', 'theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r']
    for var_name in var_names:
        sample_shape = idata.posterior[var_name].shape
        n_chains = sample_shape[0]
        n_samples = sample_shape[1]
        var_dims = sample_shape[2]
        for chain in range(n_chains):
            for dim in range(var_dims):
                all_samples = np.ravel(idata.posterior[var_name][chain,:,dim])
                first_part = all_samples[0:int(n_samples*first)]
                last_part = all_samples[n_samples-int(n_samples*last):n_samples]
                z_score = (np.mean(first_part)-np.mean(last_part)) / np.sqrt(np.var(first_part)+np.var(last_part))
                if abs(z_score) >= 1.960:
                    convergence_flag = False
                    #print("Seems like chain",chain,"has not converged in",var_name,"dimension",dim,": z_score is",z_score)
    for var_name in ['sigma_obs']:
        all_samples = np.ravel(idata.posterior[var_name])
        n_samples = len(all_samples)
        first_part = all_samples[0:int(n_samples*first)]
        last_part = all_samples[n_samples-int(n_samples*last):n_samples]
        z_score = (np.mean(first_part)-np.mean(last_part)) / np.sqrt(np.var(first_part)+np.var(last_part))
        if abs(z_score) >= 1.960:
            convergence_flag = False
            print("Seems like chain",chain,"has not converged in",var_name,"dimension",dim,": z_score is",z_score)
    if convergence_flag:
        print("All chains seem to have converged.")
    else:
        print("Seems like some chains did not converge.")
    return 0

# This is where we make sure the ii matches between X_train and train_dict: 
def get_ii_indexed_subset_dict(raw_subset_df, full_dict): # Requires raw_subset_df with reset indices
    new_dict= {}
    for ii, row in raw_subset_df.iterrows():
        patient_name = row["USUBJID"]
        new_dict[ii] = full_dict[patient_name]
    return new_dict


#####################################
# Plotting
#####################################
#treat_colordict = dict(zip(treatment_line_ids, treat_line_colors))
def plot_mprotein(patient, title, savename, PLOT_PARAMETERS=False, parameters = [], PLOT_lightchains=False, plot_pfs=False, plot_KapLam=False):
    measurement_times = patient.measurement_times
    Mprotein_values = patient.Mprotein_values
    
    fig, ax1 = plt.subplots()
    #ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='k', label="Observed M protein")
    if PLOT_lightchains:
        ax1.plot(measurement_times, Mprotein_values, linestyle='', marker='x', zorder=3, color='b', label="Observed M protein")
    if plot_pfs:
        ax1.axvline(patient.pfs, color="r", linewidth=1, linestyle="--", label="PFS")
    if plot_KapLam:
        #ax1.plot(measurement_times, patient.KappaLambdaRatio, linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
        ax1.plot(patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")

    if PLOT_PARAMETERS:
        plotting_times = np.linspace(measurement_times[0], measurement_times[-1], 80)
        treatment_history = np.array([Treatment(start=measurement_times[0], end=measurement_times[-1], id=1)])
        # Plot true M protein curves according to parameters
        plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
        # Count resistant part
        resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
        plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
        # sens
        ax1.plot(plotting_times, plotting_mprotein_values-plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=3, color='b') #, label="True M protein (total)")
        # Plot resistant M protein
        ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r") #, label="True M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k') #, label="True M protein (total)")

    ax1.set_title(title)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/L)")
    #ax1.set_ylim(bottom=0, top=85) # Not on server
    ax1.set_zorder(ax1.get_zorder()+3)
    ax1.legend()
    fig.tight_layout()
    plt.savefig(savename,dpi=300)
    plt.close()

def plot_fit_and_predictions(idata, patient_dictionary_full, N_patients_train, SAVEDIR, name, y_resolution, CLIP_MPROTEIN_TIME, CI_with_obs_noise=False, PLOT_RESISTANT=True, PLOT_TRAIN=True, clinic_view=False, p_progression=[], time_scale=1, M_scale=1):
    #######
    # Scale time back
    CLIP_MPROTEIN_TIME = CLIP_MPROTEIN_TIME * time_scale
    #######
    sample_shape = idata.posterior['psi'].shape # [chain, N_samples, dim]
    n_chains = sample_shape[0]
    N_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    # Posterior CI for train data
    if CI_with_obs_noise:
        if N_samples <= 10:
            N_rand_obs_pred_train = 10000 # Number of observation noise samples to draw for each parameter sample
        elif N_samples <= 100:
            N_rand_obs_pred_train = 1000 # Number of observation noise samples to draw for each parameter sample
        elif N_samples <= 1000:
            N_rand_obs_pred_train = 100 # Number of observation noise samples to draw for each parameter sample
        else:
            N_rand_obs_pred_train = 10 # Number of observation noise samples to draw for each parameter sample
    else:
        N_rand_obs_pred_train = 1
    print("Plotting posterior credible bands for all cases")
    for ii, patient in patient_dictionary_full.items():
        measurement_times = patient.get_measurement_times() * time_scale
        if ii < N_patients_train and not PLOT_TRAIN:
            continue
        np.random.seed(ii) # Seeding the randomness in observation noise sigma
        if clinic_view:
            time_max = CLIP_MPROTEIN_TIME + 6*28
            treatment_history = np.array([Treatment(start=1, end=CLIP_MPROTEIN_TIME+6*28, id=1)])
        else:
            time_max = find_max_time(measurement_times)
            treatment_history = patient.get_treatment_history()
        first_time = min(measurement_times[0], treatment_history[0].start)
        plotting_times = np.linspace(first_time, time_max, y_resolution) #int((measurement_times[-1]+1)*10))
        posterior_parameters = np.empty(shape=(n_chains, N_samples), dtype=object)
        predicted_y_values = np.empty(shape=(n_chains, N_samples*N_rand_obs_pred_train, y_resolution))
        predicted_y_resistant_values = np.empty_like(predicted_y_values)
        for ch in range(n_chains):
            for sa in range(N_samples):
                this_sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa])
                this_psi       = np.ravel(idata.posterior['psi'][ch,sa,ii]) * M_scale
                this_pi_r      = np.ravel(idata.posterior['pi_r'][ch,sa,ii])
                this_rho_s     = np.ravel(idata.posterior['rho_s'][ch,sa,ii]) / time_scale
                this_rho_r     = np.ravel(idata.posterior['rho_r'][ch,sa,ii]) / time_scale
                posterior_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=this_pi_r, g_r=this_rho_r, g_s=this_rho_s, k_1=0, sigma=this_sigma_obs)
                these_parameters = posterior_parameters[ch,sa]
                resistant_parameters = Parameters((these_parameters.Y_0*these_parameters.pi_r), 1, these_parameters.g_r, these_parameters.g_s, these_parameters.k_1, these_parameters.sigma)
                # Predicted total and resistant M protein
                predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, plotting_times, treatment_history)
                predicted_y_resistant_values_noiseless = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
                # Add noise and make the resistant part the estimated fraction of the observed value
                if CI_with_obs_noise:
                    for rr in range(N_rand_obs_pred_train):
                        noise_array = np.random.normal(0, this_sigma_obs, y_resolution)
                        noisy_observations = predicted_y_values_noiseless + noise_array
                        predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                        predicted_y_resistant_values[ch, N_rand_obs_pred_train*sa + rr] = predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
                else: 
                    predicted_y_values[ch, sa] = predicted_y_values_noiseless
                    predicted_y_resistant_values[ch, sa] = predicted_y_values[ch, sa] * (predicted_y_resistant_values_noiseless/(predicted_y_values_noiseless + 1e-15))
        flat_pred_y_values = np.reshape(predicted_y_values, (n_chains*N_samples*N_rand_obs_pred_train,y_resolution))
        sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
        flat_pred_resistant = np.reshape(predicted_y_resistant_values, (n_chains*N_samples*N_rand_obs_pred_train,y_resolution))
        sorted_pred_resistant = np.sort(flat_pred_resistant, axis=0)
        if ii < N_patients_train:
            IS_TEST_PATIENT = False
            if clinic_view:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_train_" + patient.name + "_clinic_view.pdf"
            else:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_train_" + patient.name + ".pdf"
        else:
            IS_TEST_PATIENT = True
            if clinic_view:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_test_" + patient.name + "_clinic_view.pdf"
            else:
                savename = SAVEDIR+name+"_CI_patient_id_"+str(ii)+"_test_" + patient.name + ".pdf"
        plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=[], PLOT_PARAMETERS=False, PLOT_TREATMENTS=False, plot_title="Posterior CI for patient "+str(ii), savename=savename, y_resolution=y_resolution, n_chains=n_chains, n_samples=N_samples, sorted_resistant_mprotein=sorted_pred_resistant, PLOT_RESISTANT=PLOT_RESISTANT, IS_TEST_PATIENT=IS_TEST_PATIENT, CLIP_MPROTEIN_TIME=CLIP_MPROTEIN_TIME, clinic_view=clinic_view, p_progression=p_progression, time_scale=time_scale, M_scale=M_scale)
    print("...done.")

def plot_posterior_local_confidence_intervals(ii, patient, sorted_local_pred_y_values, parameters=[], PLOT_PARAMETERS=False, PLOT_TREATMENTS=False, plot_title="", savename="0", y_resolution=1000, n_chains=4, n_samples=1000, sorted_resistant_mprotein=[], PLOT_MEASUREMENTS = True, PLOT_RESISTANT=True, IS_TEST_PATIENT=False, CLIP_MPROTEIN_TIME=0, clinic_view=False, p_progression=[], time_scale=1, M_scale=1, plot_pfs=False, plot_KapLam=False, print_p_progression=False, plot_CI=True):
    IS_TEST_PATIENT = True
    CLIP_MPROTEIN_TIME = 1 + 5*28
    CLIP_MPROTEIN_TIME = 1 + 8*28
    Mprotein_values = patient.get_Mprotein_values() * M_scale
    measurement_times = patient.get_measurement_times() * time_scale
    ######
    # Scale back
    patient.Mprotein_values = patient.Mprotein_values * M_scale
    patient.measurement_times = patient.measurement_times * time_scale
    for covariate_name, time_series in patient.longitudinal_data.items():
        patient.longitudinal_times[covariate_name] = patient.longitudinal_times[covariate_name] * time_scale
    ######
    if clinic_view:
        treatment_history = np.array([Treatment(start=1, end=CLIP_MPROTEIN_TIME+6*28, id=1)])
        time_max = CLIP_MPROTEIN_TIME+6*28
    else:
        treatment_history = patient.get_treatment_history()
        time_max = find_max_time(measurement_times)
    time_zero = min(treatment_history[0].start, measurement_times[0])
    plotting_times = np.linspace(time_zero, time_max, y_resolution)
    
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax1.patch.set_facecolor('none')

    if IS_TEST_PATIENT:
        ax1.axvline(CLIP_MPROTEIN_TIME, color="k", linewidth=1, linestyle="-", zorder=3.9)

    # Plot posterior confidence intervals for Resistant M protein
    # 95 % empirical confidence interval
    if PLOT_RESISTANT:
        if len(sorted_resistant_mprotein) > 0: 
            for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
                # Get index to find right value 
                lower_index = int(critical_value*sorted_resistant_mprotein.shape[0]) #n_chains*n_samples)
                upper_index = int((1-critical_value)*sorted_resistant_mprotein.shape[0]) #n_chains*n_samples)
                # index at intervals to get 95 % limit value
                lower_limits = sorted_resistant_mprotein[lower_index,:]
                upper_limits = sorted_resistant_mprotein[upper_index,:]
                ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.copper(1-critical_value), label='%3.0f %% CI, resistant M protein' % (100*(1-2*critical_value)), zorder=0+index*0.1)

    # Plot posterior confidence intervals for total M protein
    # 95 % empirical confidence interval
    color_array = ["#fbd1b4", "#f89856", "#e36209"] #["#fbd1b4", "#fab858", "#f89856", "#f67c27", "#e36209"] #https://icolorpalette.com/color/rust-orange
    if plot_CI:
        for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
            # Get index to find right value 
            lower_index = int(critical_value*sorted_local_pred_y_values.shape[0]) #n_chains*n_samples)
            upper_index = int((1-critical_value)*sorted_local_pred_y_values.shape[0]) #n_chains*n_samples)
            # index at intervals to get 95 % limit value
            lower_limits = sorted_local_pred_y_values[lower_index,:]
            upper_limits = sorted_local_pred_y_values[upper_index,:]
            shade_array = [0.7, 0.5, 0.35]
            ax1.fill_between(plotting_times, lower_limits, upper_limits, color=plt.cm.bone(shade_array[index]), label='%3.0f %% CI, total M protein' % (100*(1-2*critical_value)), zorder=1+index*0.1)

    if PLOT_PARAMETERS:
        # Plot true M protein curves according to parameters
        plotting_mprotein_values = measure_Mprotein_noiseless(parameters, plotting_times, treatment_history)
        # Count resistant part
        resistant_parameters = Parameters((parameters.Y_0*parameters.pi_r), 1, parameters.g_r, parameters.g_s, parameters.k_1, parameters.sigma)
        plotting_resistant_mprotein_values = measure_Mprotein_noiseless(resistant_parameters, plotting_times, treatment_history)
        # Plot resistant M protein
        if PLOT_RESISTANT:
            ax1.plot(plotting_times, plotting_mprotein_values - plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color='b', label="True M protein (sensitive)")
            ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color="r", label="True M protein (resistant)")
            #ax1.plot(plotting_times, plotting_resistant_mprotein_values, linestyle='--', marker='', zorder=2.9, color=plt.cm.hot(0.2), label="True M protein (resistant)")
        # Plot total M protein
        ax1.plot(plotting_times, plotting_mprotein_values, linestyle='-', marker='', zorder=3, color='k', label="True M protein (total)")
        #ax1.plot(plotting_times, plotting_mprotein_values, linestyle='--', marker='', zorder=3, color='cyan', label="True M protein (total)")

    # Plot M protein observations
    markscale = 1.5
    if PLOT_MEASUREMENTS == True:
        if IS_TEST_PATIENT:
            ax1.plot(measurement_times[measurement_times<=CLIP_MPROTEIN_TIME], Mprotein_values[measurement_times<=CLIP_MPROTEIN_TIME], linestyle='', markersize=markscale*6, marker='x', zorder=4, color='k', label="Observed M protein")
            if not clinic_view:
                # plot the unseen observations
                ax1.plot(measurement_times[measurement_times>CLIP_MPROTEIN_TIME], Mprotein_values[measurement_times>CLIP_MPROTEIN_TIME], linestyle='', markersize=markscale*6, markeredgewidth=1.7, marker='x', zorder=3.99, color='k', label="Unobserved M protein")
                ax1.plot(measurement_times[measurement_times>CLIP_MPROTEIN_TIME], Mprotein_values[measurement_times>CLIP_MPROTEIN_TIME], linestyle='', markersize=markscale*5.5, markeredgewidth=1, marker='x', zorder=4, color='whitesmoke', label="Unobserved M protein")
        else: 
            ax1.plot(measurement_times, Mprotein_values, linestyle='', markersize=markscale*6, marker='x', zorder=4, color='k', label="Observed M protein") #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]

    # Plot treatments
    if PLOT_TREATMENTS:
        plotheight = 1
        maxdrugkey = 0
        ax2 = ax1.twinx()
        for treat_index in range(len(treatment_history)):
            this_treatment = treatment_history[treat_index]
            if this_treatment.id != 0:
                treatment_duration = this_treatment.end - this_treatment.start
                if this_treatment.id > maxdrugkey:
                    maxdrugkey = this_treatment.id
                ax2.add_patch(Rectangle((this_treatment.start, this_treatment.id - plotheight/2), treatment_duration, plotheight, zorder=0, color="lightskyblue")) #color=treat_colordict[treat_line_id]))

    if plot_pfs == True and not clinic_view:
        ax1.axvline(patient.pfs, color="r", linewidth=1, linestyle="--", label="PFS") # This was never scaled
    if plot_KapLam:
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()
        try:
            Kappa_values = patient.longitudinal_data["Kappa Light Chain, Free"]
            Kappa_times = patient.longitudinal_times["Kappa Light Chain, Free"]
        except:
            Kappa_values = np.array([])
            Kappa_times = np.array([])

        try:
            Lambda_values = patient.longitudinal_data["Lambda Light Chain, Free"]
            Lambda_times = patient.longitudinal_times["Lambda Light Chain, Free"]
        except:
            Lambda_values = np.array([])
            Lambda_times = np.array([])
        
        try:
            new_ratio_values = patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free"]
            new_ratio_times = patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free"]
        except:
            new_ratio_values = np.array([])
            new_ratio_times = np.array([])
        
        if clinic_view:
            before_clip_mask = Kappa_times <= CLIP_MPROTEIN_TIME
            ax3.plot(Kappa_times[before_clip_mask], Kappa_values[before_clip_mask], linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(0.9), label="Kappa Light Chain SDTM")
            before_clip_mask = Lambda_times <= CLIP_MPROTEIN_TIME
            ax4.plot(Lambda_times[before_clip_mask], Lambda_values[before_clip_mask], linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(1.0), label="Lambda Light Chain SDTM")
            before_clip_mask = Lambda_times <= CLIP_MPROTEIN_TIME
            ax1.plot(new_ratio_times[before_clip_mask], new_ratio_values[before_clip_mask], linestyle='--', marker='x', zorder=2.1, color=plt.cm.viridis(0.8), label="Kappa/Lambda ratio SDTM")
            before_clip_mask = patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"] <= CLIP_MPROTEIN_TIME
            ax1.plot(patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"][before_clip_mask], patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"][before_clip_mask], linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
            plot_title = plot_title + " : " + patient.name
        else:
            #ax1.plot(measurement_times, patient.KappaLambdaRatio, linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
            ax3.plot(Kappa_times, Kappa_values, linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(0.9), label="Kappa Light Chain SDTM")
            ax4.plot(Lambda_times, Lambda_values, linestyle='-', marker='x', zorder=2.1, color=plt.cm.viridis(1.0), label="Lambda Light Chain SDTM")
            ax1.plot(new_ratio_times, new_ratio_values, linestyle='--', marker='x', zorder=2.1, color=plt.cm.viridis(0.8), label="Kappa/Lambda ratio SDTM")
            ax1.plot(patient.longitudinal_times["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], patient.longitudinal_data["Kappa Lt Chain,Free/Lambda Lt Chain,Free (RATIO)"], linestyle='-', marker='x', zorder=2, color='g', label="Kappa/Lambda ratio")
            plot_title = plot_title + " : " + patient.name

    if print_p_progression:
        plot_title = plot_title + "\nP(progress within 6 cycles) = " + str(p_progression[ii - 137])
    ax1.set_title(plot_title)

    if clinic_view:
        latest_cycle_start = int(np.rint( ((CLIP_MPROTEIN_TIME+6*28) - 1) // 28 + 1 ))
    else:
        latest_cycle_start = int(np.rint( (max(measurement_times) - 1) // 28 + 1 ))
    tickresolution = 3
    tick_labels = [1] + [cs for cs in range(tickresolution, latest_cycle_start+1, tickresolution)]
    #step_size = 6
    ##while len(tick_labels) < min(9, latest_cycle_start):
    ##    step_size = step_size - 1
    ##    tick_labels = [1] + [cs for cs in range(1 + step_size, latest_cycle_start, step_size)]
    #if len(tick_labels) < 4:
    #    step_size = 1
    #    tick_labels = [1] + [cs for cs in range(1 + step_size, latest_cycle_start, step_size)] + [latest_cycle_start]
    if len(tick_labels) < 2:
        tick_labels = [1, latest_cycle_start]
    tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
    ax1.set_xticks(tick_positions, tick_labels)
    ax1.set_xlabel("Cycles")
    if clinic_view:
        ax1.set_xlim(right=CLIP_MPROTEIN_TIME + 6*28)
    #cycle_data = [(mtime - 1) // 28 + 1 for mtime in measurement_times]
    #num_ticks = min(len(cycle_data), 7)
    #step_size = len(cycle_data) // num_ticks
    #tick_positions = measurement_times[::step_size]
    #tick_labels = cycle_data[::step_size]
    #tick_labels = [int(label) for label in tick_labels]
    
    #ax1.set_xticks(measurement_times, cycle_data)

    #tick_positions = measurement_times[1::28]
    #cycles = [xx for xx in tick_positions]
    #ax1.set_xticks(tick_positions, cycles)
    #ax1.set_xlabel("Days")
    ax1.set_ylabel("Serum M protein (g/L)")
    ax1.set_ylim(bottom=0, top=(1.1*max(Mprotein_values)))
    if plot_KapLam:
        ax3.set_ylabel("Kappa Light Chain, Free")
        ax3.set_ylim(bottom=0, top=((1.1*max(Kappa_values)) if len(Kappa_values)>1 else 1))
        ax4.set_ylabel("Lambda Light Chain, Free")
        ax4.set_ylim(bottom=0, top=((1.1*max(Lambda_values)) if len(Lambda_values)>1 else 1))
    #ax1.set_xlim(left=time_zero)
    if PLOT_TREATMENTS:
        ax2.set_ylabel("Treatment id for blue region")
        ax2.set_yticks([maxdrugkey])
        ax2.set_yticklabels([maxdrugkey])
        ax2.set_ylim(bottom=maxdrugkey-plotheight, top=maxdrugkey+plotheight)
        #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    ax1.set_zorder(ax1.get_zorder()+3)
    if plot_KapLam:
        ax3.set_zorder(ax1.get_zorder()+1)
        ax4.set_zorder(ax1.get_zorder()+2)
    #handles, labels = ax1.get_legend_handles_labels()
    #lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    #ax1.legend()
    #ax2.legend() # For drugs, no handles with labels found to put in legend.
    fig.tight_layout()
    plt.savefig(savename, dpi=300) #, bbox_extra_artists=(lgd), bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_posterior_traces(idata, SAVEDIR, name, psi_prior, model_name, patientwise=True, net_list=["rho_s", "rho_r", "pi_r"], INFERENCE_MODE="Full"):
    if model_name == "linear":
        print("Plotting posterior/trace plots")
        # Autocorrelation plots: 
        az.plot_autocorr(idata, var_names=["sigma_obs"])
        plt.close()

        az.plot_trace(idata, var_names=('alpha', 'beta_rho_r', 'beta_pi_r', 'omega', 'sigma_obs'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_group_parameters.pdf")
        plt.close()

        # Combined means combine the chains into one posterior. Compact means split into different subplots
        az.plot_trace(idata, var_names=('beta_rho_r'), lines=[('beta_rho_r', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-plot_posterior_uncompact_beta_rho_r.pdf")
        #plt.show()
        plt.close()

        # Combined means combine the chains into one posterior. Compact means split into different subplots
        az.plot_trace(idata, var_names=('beta_pi_r'), lines=[('beta_pi_r', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-plot_posterior_uncompact_beta_pi_r.pdf")
        #plt.show()
        plt.close()

        az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-forest_beta_rho_r.pdf")
        #plt.show()
        plt.close()
        az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-forest_beta_pi_r.pdf")
        #plt.show()
        plt.close()
        """
        az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.savefig(SAVEDIR+name+"-forest_pi_r.pdf")
        plt.tight_layout()
        #plt.show()
        plt.close()
        """
    elif model_name == "BNN":
        if "rho_s" in net_list:
            # Plot weights in_1 rho_s
            az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=False, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_s.pdf")
            plt.close()
            # Plot weights in_1 rho_s. Combined means combined chains
            az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=True, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_s_combined.pdf")
            plt.close()
            # Plot weights 2_out rho_s
            if INFERENCE_MODE == "Full":
                az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=False, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_s.pdf")
                plt.close()
                # Plot weights 2_out rho_s
                az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=True, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_s_combined.pdf")
                plt.close()

        if "rho_r" in net_list:
            # Plot weights in_1 rho_r
            az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=False, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_r.pdf")
            plt.close()
            # Plot weights in_1 rho_r
            az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=True, compact=False)
            plt.tight_layout()
            plt.savefig(SAVEDIR+name+"-_wts_in_1_rho_r_combined.pdf")
            plt.close()
            if INFERENCE_MODE == "Full":
                # Plot weights 2_out rho_r
                az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=False, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_r.pdf")
                plt.close()
                # Plot weights 2_out rho_r
                az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=True, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_out_rho_r_combined.pdf")
                plt.close()

        if "pi_r" in net_list:
                # Plot weights in_1 pi_r
                az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=False, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_in_1_pi_r.pdf")
                plt.close()
                # Plot weights in_1 pi_r
                az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=True, compact=False)
                plt.tight_layout()
                plt.savefig(SAVEDIR+name+"-_wts_in_1_pi_r_combined.pdf")
                plt.close()
                if INFERENCE_MODE == "Full":
                    # Plot weights 2_out pi_r
                    az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=False, compact=False)
                    plt.tight_layout()
                    plt.savefig(SAVEDIR+name+"-_wts_out_pi_r.pdf")
                    plt.close()
                    # Plot weights 2_out pi_r
                    az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=True, compact=False)
                    plt.tight_layout()
                    plt.savefig(SAVEDIR+name+"-_wts_out_pi_r_combined.pdf")
                    plt.close()

    elif model_name == "joint_BNN":
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out.pdf")
        plt.close()

        # Combined means combined chains
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_in_1_combined.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=True, compact=False)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_wts_out_combined.pdf")
        plt.close()

    if psi_prior=="lognormal":
        az.plot_trace(idata, var_names=('xi'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_group_parameters_xi.pdf")
        plt.close()
    # Test of exploration 
    az.plot_energy(idata)
    plt.savefig(SAVEDIR+name+"-plot_energy.pdf")
    plt.close()
    # Plot of coefficients
    az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_alpha.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_rho_s.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_rho_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_pi_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["psi"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(SAVEDIR+name+"-forest_psi.pdf")
    plt.close()
    if patientwise:
        az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True)
        plt.tight_layout()
        plt.savefig(SAVEDIR+name+"-_individual_parameters.pdf")
        plt.close()


#####################################
# PFS
#####################################

def get_true_pfs_new(patient_dictionary_test, time_scale=1, M_scale=1): # Regardless of CLIP_MPROTEIN_TIME
    N_patients_test = len(patient_dictionary_test)
    true_pfs = np.repeat(-1, repeats=N_patients_test)
    for ii, (patname, patient) in enumerate(patient_dictionary_test.items()):
        mprot = patient.Mprotein_values * M_scale
        times = patient.measurement_times * time_scale
        # Loop through measurements and find first case
        lowest_Mprotein = mprot[0]
        consecutive_above = 0
        for tt, time in enumerate(times):
            if tt == 0:
                continue
            # New criteria for pfs from protocol: 
            # 25% increase in serum M-component with an absolute increase of at least 0.5 g/dL in two consecutive measurements.
            if mprot[tt] > lowest_Mprotein*1.25 and mprot[tt] - lowest_Mprotein > 5:
                consecutive_above = consecutive_above + 1
                if consecutive_above == 2:
                    true_pfs[ii] = time
                    break
            # 50 g/L <=> 5 g/dL
            elif lowest_Mprotein >= 50 and mprot[tt] - lowest_Mprotein > 10:
                consecutive_above = consecutive_above + 1
                if consecutive_above == 2:
                    true_pfs[ii] = time
                    break
            else:
                consecutive_above = 0
            lowest_Mprotein = min(lowest_Mprotein, mprot[tt])
        # urine M-component with an absolute increase of at least 200 mg per 24 hours; 
        # <...> empty for now
        # an absolute increase in urine M-protein levels in patients without measurable serum and urine M-protein levels; 
        # <...> empty for now
        # bone marrow plasma cell percentage of at least 10%; 
        # <...> empty for now
        # development of new bone lesions or soft tissue plasmacytomas; 
        # <...> empty for now
        # or development of hypercalcemia.
        # urine M-component with an absolute increase of at least 200 mg per 24 hours; 
        # <...> empty for now
        # an absolute increase in urine M-protein levels in patients without measurable serum and urine M-protein levels; 
        # <...> empty for now
        # bone marrow plasma cell percentage of at least 10%; 
        # <...> empty for now
        # development of new bone lesions or soft tissue plasmacytomas; 
        # <...> empty for now
        # or development of hypercalcemia.
    #print("True PFS\n", true_pfs)
    #print("Average PFS among actual PFS", np.mean(true_pfs[true_pfs>0]))
    #print("Std PFS among actual PFS", np.std(true_pfs[true_pfs>0]))
    #print("Median PFS among actual PFS", np.median(true_pfs[true_pfs>0]))
    #print("Max PFS among actual PFS", np.max(true_pfs[true_pfs>0]))
    #print("Min PFS among actual PFS", np.min(true_pfs[true_pfs>0]))
    return true_pfs

def predict_PFS_new(idata, patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon, CI_with_obs_noise=False, y_resolution=100, time_scale=1, M_scale=1):
    #######
    # Scale time back
    CLIP_MPROTEIN_TIME = CLIP_MPROTEIN_TIME * time_scale
    end_of_prediction_horizon = end_of_prediction_horizon * time_scale
    #######
    sample_shape = idata.posterior['psi'].shape # [chain, n_samples, dim]
    n_chains = sample_shape[0]
    n_samples = sample_shape[1]
    var_dimensions = sample_shape[2] # one per patient
    # Posterior CI for train data
    if CI_with_obs_noise:
        if n_samples <= 10:
            N_rand_obs_pred_train = 10000 # Number of observation noise samples to draw for each parameter sample
        elif n_samples <= 100:
            N_rand_obs_pred_train = 1000 # Number of observation noise samples to draw for each parameter sample
        elif n_samples <= 1000:
            N_rand_obs_pred_train = 100 # Number of observation noise samples to draw for each parameter sample
        else:
            N_rand_obs_pred_train = 10 # Number of observation noise samples to draw for each parameter sample
    else:
        N_rand_obs_pred_train = 1
    N_patients_test = len(patient_dictionary_full) - N_patients_train
    p_progression = [] #np.repeat(-1, repeats=N_patients_test)
    ii_test = 0
    for ii, patient in patient_dictionary_full.items():
        ######
        measurement_times = patient.get_measurement_times() * time_scale
        Mprotein_values = patient.Mprotein_values * M_scale
        if ii >= N_patients_train:
            if ii == N_patients_train:
                print("In predict_PFS_new, finding p(progression) for ii =", ii, "and the rest...")
            np.random.seed(ii) # Seeding the randomness in observation noise sigma
            # Find the lowest limit to qualify as progression
            lowest_Mprotein = min(Mprotein_values[measurement_times <= CLIP_MPROTEIN_TIME])
            pfs_threshold = max(1.25*lowest_Mprotein, lowest_Mprotein + 0.5)
            # Removed this. This is just to save time. We still want the estimate for plots even though we cannot compare with the truth
            #times = measurement_times[measurement_times > CLIP_MPROTEIN_TIME]
            #if len(times) < 1:
            #    p_progression.append(-2)
            #    ii_test = ii_test + 1
            #else: 
            pat_hist = patient.treatment_history[0]
            treatment_history = [Treatment(pat_hist.start, end_of_prediction_horizon, pat_hist.id)]
            eval_times = np.linspace(CLIP_MPROTEIN_TIME, end_of_prediction_horizon, y_resolution) #find_max_time(times), y_resolution)
            posterior_parameters = np.empty(shape=(n_chains, n_samples), dtype=object)
            predicted_y_values = np.empty(shape=(n_chains, n_samples*N_rand_obs_pred_train, y_resolution))
            for ch in range(n_chains):
                for sa in range(n_samples):
                    this_sigma_obs = np.ravel(idata.posterior['sigma_obs'][ch,sa]) * M_scale
                    this_psi       = np.ravel(idata.posterior['psi'][ch,sa,ii]) * M_scale
                    this_pi_r      = np.ravel(idata.posterior['pi_r'][ch,sa,ii])
                    this_rho_s     = np.ravel(idata.posterior['rho_s'][ch,sa,ii]) / time_scale
                    this_rho_r     = np.ravel(idata.posterior['rho_r'][ch,sa,ii]) / time_scale
                    posterior_parameters[ch,sa] = Parameters(Y_0=this_psi, pi_r=this_pi_r, g_r=this_rho_r, g_s=this_rho_s, k_1=0, sigma=this_sigma_obs)
                    these_parameters = posterior_parameters[ch,sa]
                    predicted_y_values_noiseless = measure_Mprotein_noiseless(these_parameters, eval_times, treatment_history)
                    #print("Y_0", this_psi, "pi_r",this_pi_r, "g_r",this_rho_r, "g_s",this_rho_s)
                    #print("predicted_y_values_noiseless", predicted_y_values_noiseless)
                    #print(np.sum(predicted_y_values_noiseless >= pfs_threshold))
                    # Add noise and make the resistant part the estimated fraction of the observed value
                    if CI_with_obs_noise:
                        for rr in range(N_rand_obs_pred_train):
                            noise_array = np.random.normal(0, this_sigma_obs, y_resolution)
                            noisy_observations = predicted_y_values_noiseless + noise_array
                            predicted_y_values[ch, N_rand_obs_pred_train*sa + rr] = np.array([max(0, value) for value in noisy_observations]) # 0 threshold
                    else: 
                        predicted_y_values[ch, sa] = predicted_y_values_noiseless
            # For each time point after clip, p(progression) is the percentage of trajectories above this point
            #print("sum(predicted_y_values < 0.0001)", np.sum(predicted_y_values < 0.0001))
            N_predictions = n_chains*n_samples*N_rand_obs_pred_train
            flat_pred_y_values = np.reshape(predicted_y_values, (N_predictions, y_resolution))
            sorted_local_pred_y_values = np.sort(flat_pred_y_values, axis=0)
            p_progression_each_time = [np.sum(sorted_local_pred_y_values[:,tt] >= pfs_threshold)/N_predictions for tt in range(y_resolution)]
            p_progression.append(np.max(p_progression_each_time))
            ii_test = ii_test + 1
    assert int(ii_test) == N_patients_test
    return p_progression

def predict_PFS_velocity_model(patient_dictionary_full, N_patients_train, CLIP_MPROTEIN_TIME, end_of_prediction_horizon, time_scale=1, M_scale=1):
    #######
    # Scale time back
    CLIP_MPROTEIN_TIME = CLIP_MPROTEIN_TIME * time_scale
    end_of_prediction_horizon = end_of_prediction_horizon * time_scale
    #######
    N_patients_test = len(patient_dictionary_full) - N_patients_train
    p_progression = []
    ii_test = 0
    for ii, patient in patient_dictionary_full.items():
        measurement_times = patient.get_measurement_times() * time_scale
        Mprotein_values = patient.Mprotein_values * M_scale
        if ii >= N_patients_train:
            # Find the lowest limit to qualify as progression
            lowest_Mprotein = min(Mprotein_values[measurement_times <= CLIP_MPROTEIN_TIME])
            prev_mprot = Mprotein_values[measurement_times <= CLIP_MPROTEIN_TIME]
            prev_times = measurement_times[measurement_times <= CLIP_MPROTEIN_TIME]
            pfs_threshold = max(1.25*lowest_Mprotein, lowest_Mprotein + 0.5)
            #print("lowest_Mprotein", lowest_Mprotein)
            #print("prev_mprot", prev_mprot)
            #print("prev_times", prev_times)
            #print("pfs_threshold", pfs_threshold)
            times = measurement_times[measurement_times > CLIP_MPROTEIN_TIME]
            if len(prev_mprot) < 1: # no measurements before
                p_progression.append(-3)
                ii_test = ii_test + 1
            elif len(times) < 1 : # no measurements after
                p_progression.append(-2)
                ii_test = ii_test + 1
            else: 
                # Each measurement can have some observation noise with std 0.25
                N_predictions = 1000
                m_std = 0.5
                np.random.seed(0)
                if len(prev_mprot) < 2: # one measurement only
                    # Each measurement can have some observation noise with std 0.25
                    noise_only_one = np.random.normal(loc=0, scale=m_std, size=N_predictions)
                    m_fonly_one = prev_mprot[-1] + noise_only_one
                    #print("m_velocity", m_velocity)
                    predicted_mprot_endtime = np.array([max(0, value) for value in m_fonly_one]) # 0 threshold
                    p_progression_this_patient = np.sum(predicted_mprot_endtime >= pfs_threshold) / N_predictions
                    p_progression.append(p_progression_this_patient)
                else: # at least two measurements and we can get a velocity
                    noise_first = np.random.normal(loc=0, scale=m_std, size=N_predictions)
                    noise_second = np.random.normal(loc=0, scale=m_std, size=N_predictions)
                    m_first = prev_mprot[-2] + noise_first
                    m_second = prev_mprot[-1] + noise_second
                    m_velocity = (m_second - m_first) / (prev_times[-1] - prev_times[-2])
                    #print("m_velocity", m_velocity)
                    predicted_mprot_endtime = (m_second) + m_velocity * (end_of_prediction_horizon - prev_times[-1])
                    predicted_mprot_endtime = np.array([max(0, value) for value in predicted_mprot_endtime]) # 0 threshold
                    p_progression_this_patient = np.sum(predicted_mprot_endtime >= pfs_threshold) / N_predictions
                    p_progression.append(p_progression_this_patient)
                    #print("Patient ii", ii, "p_progression_this_patient:", p_progression_this_patient)
                ii_test = ii_test + 1
    assert int(ii_test) == N_patients_test
    #print(p_progression)
    return p_progression

def find_max_time(measurement_times):
    # Plot until last measurement time (last in array, or first nan in array)
    if np.isnan(measurement_times).any():
        last_time_index = np.where(np.isnan(measurement_times))[0][0] -1 # Last non-nan index
    else:
        last_time_index = -1
    return int(measurement_times[last_time_index])
