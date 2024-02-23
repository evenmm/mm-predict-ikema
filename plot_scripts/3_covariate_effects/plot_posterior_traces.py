# -*- coding: utf-8 -*-
# 29.06.2023 Even Moa Myklebust

import matplotlib.pyplot as plt
import arviz as az

def plot_posterior_traces(idata, PLOT_DIR, name, psi_prior, model_name, patientwise=True, net_list=["rho_s", "rho_r", "pi_r"], INFERENCE_MODE="Full", covariate_effects_on_rho_r=True, longitud_covariate_names=[]):
    print("plot_posterior_traces for", model_name)
    if model_name == "linear":
        print("Plotting posterior/trace plots")
        # Autocorrelation plots: 
        az.plot_autocorr(idata, var_names=["sigma_obs"])
        plt.close()

        az.plot_trace(idata, var_names=('alpha', 'omega', 'sigma_obs'), combined=True, divergences=None, rug=True, legend=True)
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-_group_parameters.pdf")
        plt.close() 

        # Heavy
        # Combined means combine the chains into one posterior. Compact means split into different subplots
        #if covariate_effects_on_rho_r:
        #    az.plot_trace(idata, var_names=('beta_rho_r'), lines=[('beta_rho_r', {}, [0])], combined=False, compact=False, divergences=None, rug=True, legend=True)
        #    #plt.tight_layout()
        #    plt.savefig(PLOT_DIR+name+"-plot_posterior_uncompact_beta_rho_r.pdf")
        #    #plt.show()
        #    plt.close()
        ## Combined means combine the chains into one posterior. Compact means split into different subplots
        #az.plot_trace(idata, var_names=('beta_pi_r'), lines=[('beta_pi_r', {}, [0])], combined=False, compact=False, divergences=None, rug=True, legend=True)
        ##plt.tight_layout()
        #plt.savefig(PLOT_DIR+name+"-plot_posterior_uncompact_beta_pi_r.pdf")
        ##plt.show()
        #plt.close()

        #if len(longitud_covariate_names) > 0:
        #    # Forest for longitudinal covariates alone?

        if covariate_effects_on_rho_r:
            if len(longitud_covariate_names) > 0:
                az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True, figsize=(80,80))
            else:
                az.plot_forest(idata, var_names=["beta_rho_r"], combined=True, hdi_prob=0.95, r_hat=True, figsize=(40,20))
            #plt.tight_layout()
            plt.savefig(PLOT_DIR+name+"-forest_beta_rho_r.pdf")
            #plt.show()
            plt.close()

        if len(longitud_covariate_names) > 0:
            az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True, figsize=(80,80))
        else:
            az.plot_forest(idata, var_names=["beta_pi_r"], combined=True, hdi_prob=0.95, r_hat=True, figsize=(40,20))
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-forest_beta_pi_r.pdf")
        #plt.show()
        plt.close()
        """
        az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
        plt.savefig(PLOT_DIR+name+"-forest_pi_r.pdf")
        #plt.tight_layout()
        plt.show()
        plt.close()
        """
    elif model_name == "BNN":
        if "rho_s" in net_list:
            # Plot weights in_1 rho_s
            az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=False, compact=False, divergences=None, rug=True, legend=True)
            #plt.tight_layout()
            plt.savefig(PLOT_DIR+name+"-_wts_in_1_rho_s.pdf")
            plt.close()
            # Plot weights in_1 rho_s. Combined means combined chains
            az.plot_trace(idata, var_names=('weights_in_rho_s'), combined=True, compact=False, divergences=None, rug=True, legend=True)
            #plt.tight_layout()
            plt.savefig(PLOT_DIR+name+"-_wts_in_1_rho_s_combined.pdf")
            plt.close()
            # Plot weights 2_out rho_s
            if INFERENCE_MODE == "Full":
                az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=False, compact=False, divergences=None, rug=True, legend=True)
                #plt.tight_layout()
                plt.savefig(PLOT_DIR+name+"-_wts_out_rho_s.pdf")
                plt.close()
                # Plot weights 2_out rho_s
                az.plot_trace(idata, var_names=('weights_out_rho_s'), combined=True, compact=False, divergences=None, rug=True, legend=True)
                #plt.tight_layout()
                plt.savefig(PLOT_DIR+name+"-_wts_out_rho_s_combined.pdf")
                plt.close()

        if "rho_r" in net_list:
            # Plot weights in_1 rho_r
            az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=False, compact=False, divergences=None, rug=True, legend=True)
            #plt.tight_layout()
            plt.savefig(PLOT_DIR+name+"-_wts_in_1_rho_r.pdf")
            plt.close()
            # Plot weights in_1 rho_r
            az.plot_trace(idata, var_names=('weights_in_rho_r'), combined=True, compact=False, divergences=None, rug=True, legend=True)
            #plt.tight_layout()
            plt.savefig(PLOT_DIR+name+"-_wts_in_1_rho_r_combined.pdf")
            plt.close()
            if INFERENCE_MODE == "Full":
                # Plot weights 2_out rho_r
                az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=False, compact=False, divergences=None, rug=True, legend=True)
                #plt.tight_layout()
                plt.savefig(PLOT_DIR+name+"-_wts_out_rho_r.pdf")
                plt.close()
                # Plot weights 2_out rho_r
                az.plot_trace(idata, var_names=('weights_out_rho_r'), combined=True, compact=False, divergences=None, rug=True, legend=True)
                #plt.tight_layout()
                plt.savefig(PLOT_DIR+name+"-_wts_out_rho_r_combined.pdf")
                plt.close()

        if "pi_r" in net_list:
                # Plot weights in_1 pi_r
                az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=False, compact=False, divergences=None, rug=True, legend=True)
                #plt.tight_layout()
                plt.savefig(PLOT_DIR+name+"-_wts_in_1_pi_r.pdf")
                plt.close()
                # Plot weights in_1 pi_r
                az.plot_trace(idata, var_names=('weights_in_pi_r'), combined=True, compact=False, divergences=None, rug=True, legend=True)
                #plt.tight_layout()
                plt.savefig(PLOT_DIR+name+"-_wts_in_1_pi_r_combined.pdf")
                plt.close()
                if INFERENCE_MODE == "Full":
                    # Plot weights 2_out pi_r
                    az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=False, compact=False, divergences=None, rug=True, legend=True)
                    #plt.tight_layout()
                    plt.savefig(PLOT_DIR+name+"-_wts_out_pi_r.pdf")
                    plt.close()
                    # Plot weights 2_out pi_r
                    az.plot_trace(idata, var_names=('weights_out_pi_r'), combined=True, compact=False, divergences=None, rug=True, legend=True)
                    #plt.tight_layout()
                    plt.savefig(PLOT_DIR+name+"-_wts_out_pi_r_combined.pdf")
                    plt.close()

    elif model_name == "joint_BNN":
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=False, compact=False, divergences=None, rug=True, legend=True)
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-_wts_in_1.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=False, compact=False, divergences=None, rug=True, legend=True)
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-_wts_out.pdf")
        plt.close()

        # Combined means combined chains
        # Plot weights in_1
        az.plot_trace(idata, var_names=('weights_in'), combined=True, compact=False, divergences=None, rug=True, legend=True)
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-_wts_in_1_combined.pdf")
        plt.close()

        # Plot weights 2_out
        az.plot_trace(idata, var_names=('weights_out'), combined=True, compact=False, divergences=None, rug=True, legend=True)
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-_wts_out_combined.pdf")
        plt.close()

    if psi_prior=="lognormal":
        az.plot_trace(idata, var_names=('xi'), combined=True, divergences=None, rug=True, legend=True)
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-_group_parameters_xi.pdf")
        plt.close()
    # Test of exploration 
    az.plot_energy(idata)
    plt.savefig(PLOT_DIR+name+"-plot_energy.pdf")
    plt.close()
    # Plot of coefficients
    az.plot_forest(idata, var_names=["alpha"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(PLOT_DIR+name+"-forest_alpha.pdf")
    plt.close()
    """
    az.plot_forest(idata, var_names=["rho_s"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(PLOT_DIR+name+"-forest_rho_s.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["rho_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(PLOT_DIR+name+"-forest_rho_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["pi_r"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(PLOT_DIR+name+"-forest_pi_r.pdf")
    plt.close()
    az.plot_forest(idata, var_names=["psi"], combined=True, hdi_prob=0.95, r_hat=True)
    plt.savefig(PLOT_DIR+name+"-forest_psi.pdf")
    plt.close()
    if patientwise:
        az.plot_trace(idata, var_names=('theta_rho_s', 'theta_rho_r', 'theta_pi_r', 'rho_s', 'rho_r', 'pi_r'), combined=True, divergences=None, rug=True, legend=True)
        #plt.tight_layout()
        plt.savefig(PLOT_DIR+name+"-_individual_parameters.pdf")
        plt.close()
    """
