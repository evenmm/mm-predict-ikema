# -*- coding: utf-8 -*-
# 29.06.2023 Even Moa Myklebust
import matplotlib.pyplot as plt
from plot_posterior_traces import plot_posterior_traces
import arviz as az
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# 1 load data

# 2 Plot traces with covariate effects

# 3 Look at links between baseline covariates and high/low values of model parameter: t tests

def predictive_inference():
    dataset = "IKEMA" # "IKEMA"  "ICARIA"  "ICA_and_IKE"
    df_arm="ALL"
    interactions = False
    N_patients = 229

    # Get parameter estimates by joint NLME Bayesian inference
    LOAD_DIR="./binaries/"
    PLOT_DIR="./plots/"
    psi_prior = "lognormal"
    likelihood_model = "original" # "Kevins_second" "original"
    model_rho_r_dependancy_on_rho_s = False
    covariate_effects_on_rho_r = True
    include_radiological_relapse = False
    N_samples = 3000
    N_tuning = 3000

    extend_covariates = True
    include_mrd = True

    PLOT_DATA_and_PFS = False
    PLOT_INDIVIDUAL_FIT = False
    PLOT_TRACES = True #False #True
    clinic_view = False

    load_X_matrices = True
    nlme_no_covariate_inference = False

    ADADELTA = True
    advi_iterations = 15_000
    n_init_advi = 6_000
    n_chains = 4
    CORES = 4
    FUNNEL_REPARAMETRIZATION = False
    target_accept = 0.99
    y_resolution = 80 # Number of timepoints to evaluate the posterior of y in
    # First set clip time
    # Full inference for each clip time with their own "fit" dictionary
    P = 44
    #name = dataset+"_"+df_arm+"_"+likelihood_model+"_common_fit_"+str(N_patients)+"_patients_"
    name = dataset+"_"+df_arm+"_"+likelihood_model+str(N_patients)+"_patients_"+str(N_samples)+"_P_"+str(P)+"_samples__cov_effects_rhor_"+str(covariate_effects_on_rho_r)+"_extension_"+str(extend_covariates)+"_mrd_"+str(include_mrd)
    print("name:", name)

    longitud_covariate_names = []
    try:
        print("Trying to load idata_LIN")
        picklefile = open(LOAD_DIR+name+"_individual_model_idata_pickle_LIN", "rb")
        idata_LIN = pickle.load(picklefile)
        picklefile.close()
        print("Loaded idata_LIN")
    except:
        raise ValueError("Failed to load idata")
    # 3 Plot traces wiht covariate effects
    #if PLOT_INDIVIDUAL_FIT: # Can't plot because no patient dictionary
    #    plot_fit_and_predictions(idata_LIN, patient_dictionary_full, N_patients_train, PLOT_DIR, name+"_LIN", y_resolution, scaled_CLIP_MPROTEIN_TIME, CI_with_obs_noise=False, PLOT_RESISTANT=True, PLOT_TRAIN=True)
    if PLOT_TRACES:
        plot_posterior_traces(idata_LIN, PLOT_DIR, name, psi_prior, "linear", patientwise=False, longitud_covariate_names=longitud_covariate_names)
    # Plot rho_s to rho_r link
    if model_rho_r_dependancy_on_rho_s:
        az.plot_trace(idata_LIN, var_names=('coef_rho_s'), lines=[('coef_rho_s', {}, [0])], combined=False, compact=False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-plot_posterior_uncompact_coef_rho_s_LIN.pdf")
        #plt.show()
        plt.close()
    """
    # 4 Compare survival PFS for hi and low model parameters
    #### Survival analysis on high/low model parameters vs PFS ####
    # event variables for pfs are made above 
    # Here we take treatment from X, then extract median rho rho pi values from idata_LIN
    Treat_is_Kd = X_raw["Treat_is_Kd"].tolist()
    print("Treat_is_Kd", Treat_is_Kd)
    """
    try:
        median_rho_s = np.loadtxt(LOAD_DIR+name+"_median_rho_s.csv", dtype="float", delimiter = ",")
        median_rho_r = np.loadtxt(LOAD_DIR+name+"_median_rho_r.csv", dtype="float", delimiter = ",")
        median_pi_r = np.loadtxt(LOAD_DIR+name+"_median_pi_r.csv", dtype="float", delimiter = ",")
        var_rho_s = np.loadtxt(LOAD_DIR+name+"_var_rho_s.csv", dtype="float", delimiter = ",")
        var_rho_r = np.loadtxt(LOAD_DIR+name+"_var_rho_r.csv", dtype="float", delimiter = ",")
        var_pi_r = np.loadtxt(LOAD_DIR+name+"_var_pi_r.csv", dtype="float", delimiter = ",")
        """
        # Reorder estimates to match the order of X_train + X_test
        name_index = np.genfromtxt(LOAD_DIR+name+"_name_index.csv", dtype="str")
        usubjid_index_map = {usubjid : index for index, usubjid in enumerate(X_raw["USUBJID"])}
        sorted_indices = [usubjid_index_map[usubjid] for usubjid in name_index]
        median_rho_s = median_rho_s[sorted_indices]
        median_rho_r = median_rho_r[sorted_indices]
        median_pi_r = median_pi_r[sorted_indices]
        var_rho_s = var_rho_s[sorted_indices]
        var_rho_r = var_rho_r[sorted_indices]
        var_pi_r = var_pi_r[sorted_indices]
        #print("Check if assert holds:")
        #assert (name_index[sorted_indices] == X_raw["USUBJID"].tolist()).all()
        """
        print("Loaded mean and variances")
    except: 
        ValueError("Failed to load mean and variances of parameter estimates")
        """
        print("Getting medians and variances from posterior...")
        print("Getting rho rho pi")
        sample_shape = idata_LIN.posterior["sigma_obs"].shape
        n_chains = sample_shape[0]
        n_samples = sample_shape[1]
        median_rho_s = np.zeros(N_patients)
        var_rho_s = np.zeros(N_patients)
        median_rho_r = np.zeros(N_patients)
        var_rho_r = np.zeros(N_patients)
        median_pi_r = np.zeros(N_patients)
        var_pi_r = np.zeros(N_patients)
        args = [(sample_shape, ii, idata_LIN) for ii in range(N_patients)]
        for ii, elem in enumerate(args):
            print("ii:", ii)
            median_rho_s[ii], var_rho_s[ii], median_rho_r[ii], var_rho_r[ii], median_pi_r[ii], var_pi_r[ii] = get_sample_weights(elem)
        print("Done sampling, now saving")
        np.savetxt(LOAD_DIR+name+"_median_rho_s.csv", median_rho_s, delimiter = ",")
        np.savetxt(LOAD_DIR+name+"_median_rho_r.csv", median_rho_r, delimiter = ",")
        np.savetxt(LOAD_DIR+name+"_median_pi_r.csv", median_pi_r, delimiter = ",")
        np.savetxt(LOAD_DIR+name+"_var_rho_s.csv", var_rho_s, delimiter = ",")
        np.savetxt(LOAD_DIR+name+"_var_rho_r.csv", var_rho_r, delimiter = ",")
        np.savetxt(LOAD_DIR+name+"_var_pi_r.csv", var_pi_r, delimiter = ",")
        name_index = [patient.name for ii, patient in patient_dictionary_full.items()]
        np.savetxt(LOAD_DIR+name+"_name_index.csv", name_index, fmt="%s")
        assert name_index == X_raw["USUBJID"].tolist()
        """

    """
    # Save for Survival analysis in R
    lists_to_save = [pfs_times, pfs_censoreds, pfs_eventdescs, Treat_is_Kd, median_rho_r.tolist(), median_rho_s.tolist(), median_pi_r.tolist()]
    list_names = ["pfs_times", "pfs_censoreds", "pfs_eventdescs", "Treat_is_Kd", "rho_r", "rho_s", "pi_r"]
    import json
    for lst, lst_name in zip(lists_to_save, list_names):
        print(lst_name)
        print(type(lst))

    print("Treat_is_Kd:", Treat_is_Kd)

    for lst, lst_name in zip(lists_to_save, list_names):
        with open(f'{lst_name}.json', 'w') as jsonfile:
            json.dump(lst, jsonfile)
    """
    ########################################
    # 5 Look at links between baseline covariates and high/low values of model parameter: t tests
    # ttest t-test Bonferroni Benjamini Hochberg BH-corrected corrected t-tests
    # T test for correlation with covariates
    """ # Can't look at links between covariates without covariates
    WHICH_TEST = "ttest"
    if WHICH_TEST == "ttest":
        print("T tests, comparing averages")
    elif WHICH_TEST == "utest":
        print("U tests, comparing medians")
    def find_columns_with_two_unique_values(df):
        result = []
        unique_values_result = []
        for column in df.columns:
            #unique_values = sorted(df[column].unique()) # There are both str and float
            unique_values = df[column].unique()
            if len(unique_values) == 2:
                result.append(column)
                unique_values_result.append(unique_values)
        return result, unique_values_result
    columns_with_two_unique_values, unique_values_array = find_columns_with_two_unique_values(X_raw)
    triplet = [median_rho_s, median_rho_r, median_pi_r]
    striplet = ["median_rho_s", "median_rho_r", "median_pi_r"]
    p_values = []
    import scipy as sp 
    for kk, parameter in enumerate(triplet):
        parameter_name = striplet[kk]
        from scipy.stats import probplot
        #fig, ax = plt.subplots()
        #res = probplot(abs(parameter), dist="norm", plot=ax)
        #fig, ax = plt.subplots()
        #res = probplot(np.log(abs(parameter)), dist="norm", plot=ax)
        #fig, ax = plt.subplots()
        #probplot(np.sqrt(abs(parameter)), dist="norm", plot=ax)
        #plt.show()
        if parameter_name == "median_rho_s":
            pass
            #parameter = np.sqrt(abs(parameter))
        elif parameter_name == "median_rho_r":
            pass
            #parameter = np.log(abs(parameter))
        elif parameter_name == "median_pi_r":
            pass
            #parameter = np.sqrt(abs(parameter))
        # For the binary covariates only
        for jj, covariate in enumerate(columns_with_two_unique_values):
            unique_values = unique_values_array[jj]
            subset_with_no_nan = X_raw.dropna(subset=[covariate])
            indices_a = subset_with_no_nan.loc[subset_with_no_nan[covariate] == unique_values[0]].index
            indices_b = subset_with_no_nan.loc[subset_with_no_nan[covariate] == unique_values[1]].index
            par_a = parameter[indices_a]
            par_b = parameter[indices_b]
            if min(len(par_a), len(par_b)) > 1:
                if WHICH_TEST == "ttest":
                    tval, pval = sp.stats.ttest_ind(par_a, par_b)
                elif WHICH_TEST == "utest":
                    u_res = sp.stats.mannwhitneyu(par_a, par_b)
                    tval, pval = u_res.statistic, u_res.pvalue
                if pval < 0.05: 
                    #print(parameter_name, ",", covariate, ":", "pval", pval, "Means", np.mean(par_a), "(", len(par_a), ")", "and", np.mean(par_b), "(", len(par_b), ")")
                    pass
                if np.isnan(pval):
                    print("Nan for", par_a, par_b)
                else:
                    p_values.append(pval)
    p_values_sorted = sorted(p_values)
    #print("p_values_sorted", p_values_sorted)
    print("Benjamini Hochberg")
    controlled_fdr = 0.25
    N_p_tests = 3 * len(columns_with_two_unique_values)
    #assert N_p_tests == len(p_values_sorted)
    bh_thresholds = [(ii+1)/len(p_values_sorted)*controlled_fdr for ii, elem in enumerate(p_values_sorted)]
    #print("bh_thresholds", bh_thresholds)
    last_index=None
    for ii, (a_val, b_val) in enumerate(zip(p_values_sorted, bh_thresholds)):
        if a_val < b_val:
            last_index = ii
    #print("last_index", last_index)
    all_significants = [[],[],[]]
    if last_index is not None:
        bh_sign_pvalues = p_values_sorted[:last_index+1]
        print("BH-corrected significant p-values:")
        print(bh_sign_pvalues)
        for kk, parameter in enumerate(triplet):
            parameter_name = striplet[kk]
            from scipy.stats import probplot
            #fig, ax = plt.subplots()
            #res = probplot(abs(parameter), dist="norm", plot=ax)
            #fig, ax = plt.subplots()
            #res = probplot(np.log(abs(parameter)), dist="norm", plot=ax)
            #fig, ax = plt.subplots()
            #probplot(np.sqrt(abs(parameter)), dist="norm", plot=ax)
            plt.show()
            if parameter_name == "median_rho_s":
                pass
                #parameter = np.sqrt(abs(parameter))
            elif parameter_name == "median_rho_r":
                pass
                #parameter = np.log(abs(parameter))
            elif parameter_name == "median_pi_r":
                pass
                #parameter = np.sqrt(abs(parameter))
            # For the binary covariates only
            for jj, covariate in enumerate(columns_with_two_unique_values):
                unique_values = unique_values_array[jj]
                subset_with_no_nan = X_raw.dropna(subset=[covariate])
                indices_a = subset_with_no_nan.loc[subset_with_no_nan[covariate] == unique_values[0]].index
                indices_b = subset_with_no_nan.loc[subset_with_no_nan[covariate] == unique_values[1]].index
                par_a = parameter[indices_a]
                par_b = parameter[indices_b]
                if min(len(par_a), len(par_b)) > 1:
                    tval, pval = sp.stats.ttest_ind(par_a, par_b)
                    if pval <= max(bh_sign_pvalues):
                        all_significants[kk].append(covariate)
                        print(parameter_name, ",", covariate, ":", "pval", pval, "Means", np.mean(par_a), "(", len(par_a), "patients with value for value", unique_values[0], ")", "and", np.mean(par_b), "(", len(par_b), "patients with value for value", unique_values[1], ")")
        if WHICH_TEST == "ttest":
            print("All T-test significants printed")
        elif WHICH_TEST == "utest":
            print("All U-test significants printed")
    else:
        bh_sign_pvalues = []
        print("No covariates found significant by Benjamini-Hochberg (or 0.5). Continuing with default significant ones from seed=0")

    # "ttest-like"
    # Plot comparison of values 0 and 1 for significant binary covariates
    for kk, parameter in enumerate(triplet):
        parameter_name = striplet[kk]
        significant_columns = all_significants[kk]
        #if parameter_name == "median_rho_s":
        #    significant_columns = ["BL_smoke_status_2.0", "BL_smoke_status_3.0", "ISS_study_entry_3", "MM_type_at_studyentry_Ig G"]
        #elif parameter_name == "median_rho_r":
        #    significant_columns = ["MM_type_at_studyentry_Ig G", "More_than_3_prev_lines_of_therapy_1.0"]
        #elif parameter_name == "median_pi_r":
        #    significant_columns = [] #["Autologuous_transplant_1.0"]
        # For the binary covariates only
        for jj, covariate in enumerate(columns_with_two_unique_values):
            if covariate in significant_columns:
                unique_values = unique_values_array[jj]
                subset_with_no_nan = X_all.dropna(subset=[covariate])
                indices_a = subset_with_no_nan.loc[subset_with_no_nan[covariate] == unique_values[0]].index
                indices_b = subset_with_no_nan.loc[subset_with_no_nan[covariate] == unique_values[1]].index
                par_a = parameter[indices_a]
                par_b = parameter[indices_b]
                plt.figure()
                plt.plot(par_a, "o", color="b")
                plt.plot(par_b, "o", color="r")
                plt.axhline(np.mean(par_a), color="b", label=covariate+"="+str(unique_values[0]))
                plt.axhline(np.mean(par_b), color="r", label=covariate+"="+str(unique_values[1]))
                plt.xlabel("Patient id")
                plt.ylabel(parameter_name)
                plt.title(parameter_name+" by "+covariate)
                plt.legend()
                plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_"+parameter_name+"_"+covariate+".pdf", dpi=300)
                #plt.show()
                plt.close()
                #
                fig, ax1 = plt.subplots()
                averages = [par_a, par_b]
                covariate_colors = [plt.cm.magma(0.9), plt.cm.magma(0.7)]
                if WHICH_TEST == "ttest":
                    violin_params = ax1.violinplot(averages, positions=[unique_values[0],unique_values[1]], showmeans=True, showmedians=False, showextrema=False) #widths=0.3, 
                    plt.title(WHICH_TEST+": "+parameter_name+" by "+covariate+", line=average")
                elif WHICH_TEST == "utest":
                    plt.title(WHICH_TEST+": "+parameter_name+" by "+covariate+", line=median")
                    violin_params = ax1.violinplot(averages, positions=[unique_values[0],unique_values[1]], showmeans=False, showmedians=True, showextrema=False) #widths=0.3, 
                for ll, pc in enumerate(violin_params["bodies"]):
                    pc.set_facecolor(covariate_colors[ll])
                    pc.set_alpha(1)
                #    pc.set_edgecolor("black")
                #violin_params["cmedians"].set_edgecolor("black")
                #ax1.set_xticks([yy for yy in range(1,7)], labels=[str(aa) for aa in range(1,7)])
                #ax1.plot(par_a, "o", color=covariate_colors[0])
                #ax1.plot(par_b, "o", color=covariate_colors[1])
                ax1.scatter(np.repeat(unique_values[0], repeats=len(par_a)) + np.array([(0.3*kk/len(par_a)-0.15)*int(par_kk<0.06) for kk, par_kk in enumerate(par_a)]), par_a, s=3, color="k") #, color=covariate_colors[0])
                ax1.scatter(np.repeat(unique_values[1], repeats=len(par_b)) + np.array([(0.3*kk/len(par_b)-0.15)*int(par_kk<0.06) for kk, par_kk in enumerate(par_b)]), par_b, s=3, color="k") #, color=covariate_colors[1])
                ax1.set_xlabel(covariate)
                ax1.set_ylabel(parameter_name)
                ax1.set_xticks([yy for yy in range(2)], labels=[str(aa) for aa in range(2)])
                plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_"+parameter_name+"_"+covariate+"_"+WHICH_TEST+"_violin.pdf", dpi=300)
                #plt.show()
                plt.close()

    #significant_columns = [covar for sublist in significant_columns for covar in sublist]
    #significant_columns = np.array(significant_columns).flatten()
    significant_columns = all_significants[0] + all_significants[1] + all_significants[2]
    print("significant_columns:\n", significant_columns)

    #significant_columns from before: = ["BL_smoke_status_2.0", "BL_smoke_status_3.0", "ISS_study_entry_3", "MM_type_at_studyentry_Ig G"] + ["MM_type_at_studyentry_Ig G", "More_than_3_prev_lines_of_therapy_1.0"]
    #significant_columns from before: = ["Calcium_group_2.0", "ISS_revised_at_entry_3", "Treat_is_Pd_1.0"] #"Pharmacokinetic_populationis_2_1.0", 
    """

    #####################################################
    # 6 Check scatterplot correlations between parameters
    rho_r_on_rho_s_regression = False
    if rho_r_on_rho_s_regression:
        log_pi_r = np.log(0.0000001 + median_pi_r)

        # regress median_rho_r on median_rho_s
        model = LinearRegression()
        rho_s_X = [[ss] for ss in median_rho_s]
        model.fit(rho_s_X, median_rho_r)
        rho_r_pred = model.predict(rho_s_X)
        corr, pval = pearsonr(median_rho_s, median_rho_r)
        print("Correlation", corr)
        print("P value", pval)
        plt.figure()
        plt.plot(median_rho_s, rho_r_pred)
        plt.scatter(median_rho_s, median_rho_r)
        plt.xlabel("median_rho_s")
        plt.ylabel("median_rho_r")
        plt.title("Correlation " + str(corr) + " P value " + str(pval))
        plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_rho_vs_rho.pdf", dpi=300)
        plt.show()
        plt.close()

        """
        lowest_M = [min(patient.Mprotein_values) for _, patient in dict_ALL.items()]
        # regress lowest_M on median_rho_s
        model = LinearRegression()
        rho_s_X = [[ss] for ss in median_rho_s]
        model.fit(rho_s_X, lowest_M)
        lowest_M_pred = model.predict(rho_s_X)
        corr, pval = pearsonr(median_rho_s, lowest_M)
        print("Correlation", corr)
        print("P value", pval)
        plt.figure()
        plt.plot(median_rho_s, lowest_M_pred)
        plt.scatter(median_rho_s, lowest_M)
        plt.xlabel("median_rho_s")
        plt.ylabel("lowest_M")
        plt.title("Correlation " + str(corr) + " P value " + str(pval))
        plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_lowest_M_vs_rho.pdf", dpi=300)
        plt.show()
        plt.close()
        """

        # Variance in rho_s, var_... and log_pi_r against median_rho_s
        model.fit(rho_s_X, var_rho_s)
        var_rho_s_pred = model.predict(rho_s_X)
        corr, pval = pearsonr(median_rho_s, var_rho_s)
        print("Correlation var_rho_s", corr)
        print("P value var_rho_s", pval)
        plt.figure()
        plt.plot(median_rho_s, var_rho_s_pred, color="k")
        plt.scatter(median_rho_s, var_rho_s, color="k", label="var rho_s")
        plt.xlabel("median_rho_s")
        plt.ylabel("Variance of rho_s estimate")
        plt.legend()
        plt.title("Correlation " + str(corr) + " P value " + str(pval))
        plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_variances_in_rho_s_vs_rho.pdf", dpi=300)
        plt.show()
        plt.close()

        model.fit(rho_s_X, var_rho_r)
        var_rho_r_pred = model.predict(rho_s_X)
        corr, pval = pearsonr(median_rho_s, var_rho_r)
        print("Correlation var_rho_r", corr)
        print("P value var_rho_r", pval)
        plt.figure()
        plt.plot(median_rho_s, var_rho_r_pred, color="b")
        plt.scatter(median_rho_s, var_rho_r, color="b", label="var rho_r")
        plt.xlabel("median_rho_s")
        plt.ylabel("Variance of rho_r estimate")
        plt.legend()
        plt.title("Correlation " + str(corr) + " P value " + str(pval))
        plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_variances_in_rho_r_vs_rho.pdf", dpi=300)
        plt.show()
        plt.close()

        model.fit(rho_s_X, var_pi_r)
        var_pi_r_pred = model.predict(rho_s_X)
        corr, pval = pearsonr(median_rho_s, var_pi_r)
        print("Correlation var_pi_r", corr)
        print("P value var_pi_r", pval)
        plt.figure()
        plt.plot(median_rho_s, var_pi_r_pred, color="g")
        plt.scatter(median_rho_s, var_pi_r, color="g", label="Var(pi_r)")
        plt.xlabel("median_rho_s")
        plt.ylabel("Variance of pi_r estimate")
        plt.legend()
        plt.title("Correlation " + str(corr) + " P value " + str(pval))
        plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_variances_in_params_vs_rho.pdf", dpi=300)
        plt.show()
        plt.close()

        # regress log_pi_r on median_rho_s
        model = LinearRegression()
        rho_s_X = [[ss] for ss in median_rho_s]
        model.fit(rho_s_X, log_pi_r)
        pi_r_pred = model.predict(rho_s_X)
        corr, pval = pearsonr(median_rho_s, log_pi_r)
        print("Correlation", corr)
        print("P value", pval)
        plt.figure()
        plt.plot(median_rho_s, pi_r_pred)
        plt.scatter(median_rho_s, log_pi_r)
        plt.xlabel("median_rho_s")
        plt.ylabel("log_pi_r")
        plt.title("Correlation " + str(corr) + " P value " + str(pval))
        plt.savefig(PLOT_DIR+dataset+"_"+df_arm+"_pi_vs_rho.pdf", dpi=300)
        plt.show()

    """
    ########################
    # Common fit without covariates inference #
    ########################
    # NLME model without covariates 
    """
    if nlme_no_covariate_inference:
        """
        model = individual_model(patient_dictionary_full, name, N_samples, N_tuning, psi_prior=psi_prior, FUNNEL_REPARAMETRIZATION=FUNNEL_REPARAMETRIZATION, t_max_prior_scale=time_scale, likelihood_model=likelihood_model, model_rho_r_dependancy_on_rho_s=model_rho_r_dependancy_on_rho_s)
        """
        try: 
            print("Attempting to load idata...")
            picklefile = open(LOAD_DIR+name+"_individual_model_idata_pickle", "rb")
            idata = pickle.load(picklefile)
            picklefile.close()
        except:
            ValueError("Failed to load plain idata")
            """
            print("Getting idata without load:")
            with model:
                print("cores =", CORES)
                if ADADELTA:
                    print("------------------- INDEPENDENT ADVI -------------------")
                    advi = pm.ADVI()
                    #tracker = pm.callbacks.Tracker(
                    #    mean=advi.approx.mean.eval,  # callable that returns mean
                    #    std=advi.approx.std.eval,  # callable that returns std
                    #)
                    approx = advi.fit(advi_iterations, obj_optimizer=pm.adagrad_window(), obj_n_mc=25, total_grad_norm_constraint=10_000.) #, callbacks=[tracker]
        
                    print("-------------------SAMPLING-------------------")
                    # Use approx as starting point for NUTS: https://www.pymc.io/projects/examples/en/latest/variational_inference/GLM-hierarchical-advi-minibatch.html
                    scaling = approx.cov.eval()
                    sample = approx.sample(return_inferencedata=False, size=n_chains)
                    del approx
                    gc.collect()
                    start_dict = list(sample[i] for i in range(n_chains))    
                    # essentially, this is what init='advi' does
                    step = pm.NUTS(scaling=scaling, is_cov=True)
                    del sample
                    del scaling
                    gc.collect()
                    idata = pm.sample(draws=N_samples, tune=N_tuning, step=step, initvals=start_dict, chains=n_chains , cores=CORES)
                    del start_dict
                    gc.collect()
                else:
                    idata = pm.sample(draws=N_samples, tune=N_tuning, init="advi+adapt_diag", n_init=n_init_advi, random_seed=42, target_accept=target_accept, chains=n_chains, cores=CORES)
            print("Done sampling, sleep half a minute for safety")
            time.sleep(30)
            picklefile = open(LOAD_DIR+name+"_individual_model_idata_pickle", "wb")
            pickle.dump(idata, picklefile)
            picklefile.close()
            dictfile = open(LOAD_DIR+name+"_individual_model_patient_dictionary", "wb")
            pickle.dump(patient_dictionary_full, dictfile)
            dictfile.close()
            np.savetxt(LOAD_DIR+name+"_individual_model_patient_dictionary"+".csv", [patient.name for _, patient in patient_dictionary_full.items()], fmt="%s")
            """
        # 4 predictive plots for test, fit plots for train
        if PLOT_TRACES:
            plot_posterior_traces(idata, PLOT_DIR, name, psi_prior, "none", patientwise=False)
        if model_rho_r_dependancy_on_rho_s:
            # Plot rho_s to rho_r link
            az.plot_trace(idata, var_names=('coef_rho_s'), lines=[('coef_rho_s', {}, [0])], combined=False, compact=False)
            plt.tight_layout()
            plt.savefig(PLOT_DIR+name+"-plot_posterior_uncompact_coef_rho_s_indmodel.pdf")
            #plt.show()
            plt.close()
        """
        if PLOT_INDIVIDUAL_FIT:
            plot_fit_and_predictions(idata, deepcopy(patient_dictionary_full), N_patients_train, PLOT_DIR, name, y_resolution, 13, CI_with_obs_noise=False, PLOT_RESISTANT=True, PLOT_TRAIN=False, clinic_view=clinic_view, p_progression=0, time_scale=time_scale, M_scale=M_scale)
        del idata
        """
        #gc.collect()
    #del patient_dictionary_full
    #gc.collect()
    print("Done")

##########################################
#### control panel #######################
##########################################
if __name__ == "__main__":
    """
    dataset = "IKEMA" # "IKEMA"  "ICARIA"  "ICA_and_IKE"
    interactions = False
    N_patients_all = 229
    random_seed_all = 0
    # 1 load data
    # Workdir is evenmmV
    # Load patient dictionaries for both arms
    with open('./binaries_and_pickles/patient_dict_IKEMA_Kd', 'rb') as picklefile:
        patient_dict_Kd = pickle.load(picklefile)
    #N_patients_Kd, t_Kd, Y_Kd = get_N_t_Y(patient_dict_Kd)
    with open('./binaries_and_pickles/patient_dict_IKEMA_IKd', 'rb') as picklefile:
        patient_dict_IKd = pickle.load(picklefile)
    #N_patients_IKd, t_IKd, Y_IKd = get_N_t_Y(patient_dict_IKd)
    # All patients
    patient_dict_all = deepcopy(patient_dict_Kd)
    for name, patient in patient_dict_IKd.items():
        patient_dict_all[name] = patient
    N_patients_all, t_all, Y_all = get_N_t_Y(patient_dict_all)
    #del patient_dict_Kd
    #del patient_dict_IKd
    ## IKEMA
    """

    """
    print("Creating X for IKd patients:")
    X_Ikd, patient_dictionary_complete_IKd= create_X_IKEMA_folds(patient_dict_IKd, interactions, random_seed=random_seed_all)
    print("Creating X for Kd patients:")
    X_Kd, patient_dictionary_complete_Kd = create_X_IKEMA_folds(patient_dict_Kd, interactions, random_seed=random_seed_all)
    """
    
    """
    print("Creating X for All patients:")
    X_all, patient_dictionary_complete_all, raw_floats_adtte, raw_floats_adsl = create_X_IKEMA_folds(patient_dict_all)
    predictive_inference(X_all, patient_dictionary_complete_all, "IKEMA", "ALL", raw_floats_adtte, raw_floats_adsl)
    """
    predictive_inference()
