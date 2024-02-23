# -*- coding: utf-8 -*-
# 29.06.2023 Even Moa Myklebust
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as metrics

# 1 load data
# binary_progress_or_not tells if patient progressed or not
# new_p_progression tells the probability of progression predicted by the NLME model
# new_p_progression_LIN tells the probability of progression predicted by the NLME model with covariates
# new_p_progression_velo tells the probability of progression predicted by the velocity model

# 2 calculate metrics 
# AUC, accuracy, F1 score, precision, recall, specificity
# for each fold, for each clip time

# 3 Plot average metrics across the 6 folds, for each clip time


def plot_cumulative(): #X, patient_dictionary_complete):
    # Get parameter estimates by joint NLME Bayesian inference
    dataset = "IKEMA"
    df_arm = "ALL"
    SAVEDIR="./binaries/"
    PLOTDIR="./plots/"
    likelihood_model = "original" # "Kevins_second" "original"
    """
    include_radiological_relapse = False
    """
    N_patients = 229
    picklefile = open(SAVEDIR+"all_N_progressors", "rb")
    all_N_progressors = pickle.load(picklefile)
    picklefile.close()

    #n_splits = 30 # Does not matter that this matches load variables here, like it does in create_binary_fast
    #n_splits = 3
    #which_folds = range(n_splits)
    pred_window_length = 6*28

    # Works:
    which_folds = [0,1,2,4,5,6]
    #pred_window_starts = range(1+1*28, 1+12*28, 1*28)   # Works
    pred_window_starts = range(1+1*28, 1+20*28, 1*28)

    print("pred_window_starts:", pred_window_starts)

    n_splits = len(which_folds)

    predetermined_threshold = 0.03
    predetermined_threshold_velo = 0.45

    velo_threshold_80_trp_from_clip_10 = -np.inf
    nlme_threshold_80_trp_from_clip_10 = -np.inf
    LIN_threshold_80_trp_from_clip_10 = -np.inf

    threshold_dict = {
        1 : "reach 80 % TPR",
        2 : "maximize F1",
        3 : "be "+"{:.2f}".format(predetermined_threshold)+" for NLME and "+"{:.2f}".format(predetermined_threshold_velo)+" for velo",
        4 : "reach 80 % TPR for first 10, then fixed to fold 10 threshold",
        5 : "reach 80 % TPR for first 5, then fixed to fold 5 threshold",
        6 : "average of threshold that reaches 80 % TPR for 4 and 5 across folds",
        7 : "average max F1 threshold from 1-5 from 6 and up",
    }
    for t_choice in [1,2,3,4,5]: # [3]: # [1,2,3,4]:
        threshold_logic = threshold_dict[t_choice]

        best_f1_velo = np.zeros((len(pred_window_starts), n_splits))
        best_f1_nlme = np.zeros((len(pred_window_starts), n_splits))
        best_f1_LIN = np.zeros((len(pred_window_starts), n_splits))
        best_accuracy_velo = np.zeros((len(pred_window_starts), n_splits))
        best_accuracy_nlme = np.zeros((len(pred_window_starts), n_splits))
        best_accuracy_LIN = np.zeros((len(pred_window_starts), n_splits))
        best_precision_velo = np.zeros((len(pred_window_starts), n_splits))
        best_precision_nlme = np.zeros((len(pred_window_starts), n_splits))
        best_precision_LIN = np.zeros((len(pred_window_starts), n_splits))
        best_recall_velo = np.zeros((len(pred_window_starts), n_splits))
        best_recall_nlme = np.zeros((len(pred_window_starts), n_splits))
        best_recall_LIN = np.zeros((len(pred_window_starts), n_splits))
        best_specificity_velo = np.zeros((len(pred_window_starts), n_splits))
        best_specificity_nlme = np.zeros((len(pred_window_starts), n_splits))
        best_specificity_LIN = np.zeros((len(pred_window_starts), n_splits))
        all_best_threshold_velo = np.zeros((len(pred_window_starts), n_splits))
        all_best_threshold_nlme = np.zeros((len(pred_window_starts), n_splits))
        all_best_threshold_LIN = np.zeros((len(pred_window_starts), n_splits))
        auc_velo = np.zeros((len(pred_window_starts), n_splits))
        auc_nlme = np.zeros((len(pred_window_starts), n_splits))
        auc_LIN = np.zeros((len(pred_window_starts), n_splits))
        # Full inference for each clip time with their own "fit" dictionary
        for clip_index, CLIP_MPROTEIN_TIME in enumerate(pred_window_starts): 
            print("\n\nclip_index", clip_index)
            print("CLIP_MPROTEIN_TIME:", CLIP_MPROTEIN_TIME)
            end_of_prediction_horizon = CLIP_MPROTEIN_TIME + pred_window_length
            print("end_of_prediction_horizon", end_of_prediction_horizon)
            # Stratify based on relapse status
            """
            ii_complete_dict = get_ii_indexed_subset_dict(X, patient_dictionary_complete)
            relapse_label = get_relapse_label(ii_complete_dict, CLIP_MPROTEIN_TIME, end_of_prediction_horizon, include_radiological_relapse=include_radiological_relapse)
            assert len(relapse_label) == X.shape[0]

            relapse_label = np.array(relapse_label)
            print("Total N patients", len(ii_complete_dict))
            N_progressors = len(relapse_label[relapse_label == 1])
            print("Progressors", N_progressors)
            if len(relapse_label) > 0:
                proportion_progressions = N_progressors / len(relapse_label)
            else:
                proportion_progressions = 0
            print("Proportion of progressors in test fold:", proportion_progressions)
            """
            # 3 Split into fice train/test partitions, stratified by relapse_label
            #from sklearn.model_selection import StratifiedKFold
            #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            #for fold_index, (train_index, test_index) in enumerate(skf.split(X, relapse_label)):
            N_progressors = int(all_N_progressors[clip_index])
            stored_fpr_velo = []
            stored_tpr_velo = []
            stored_fpr_nlme = []
            stored_tpr_nlme = []
            stored_fpr_LIN = []
            stored_tpr_LIN = []
            stored_precision_velo = []
            stored_recall_velo = []
            stored_precision_nlme = []
            stored_recall_nlme = []
            stored_precision_LIN = []
            stored_recall_LIN = []
            # For aggregate scores across folds:
            fold_cumul_binary_progress_or_not = np.array([])
            fold_cumul_new_p_progression_velo = np.array([])
            fold_cumul_new_p_progression = np.array([])
            fold_cumul_new_p_progression_LIN = np.array([])

            # iterate folds
            for dummy_fold_index, actual_fold_index in enumerate(which_folds):
                print(f"\nFold {actual_fold_index}:")
                name = dataset+"_"+df_arm+"_"+likelihood_model+"_using_"+str(CLIP_MPROTEIN_TIME)+"_days_predict_"+str(pred_window_length)+"_ahead_"+str(N_patients)+"_patients_"+str(N_progressors)+"_progressors"+"_fold_"+str(actual_fold_index)+"_"
                picklefile = open(SAVEDIR+name+"_binary_progress_or_not", "rb")
                binary_progress_or_not = pickle.load(picklefile)
                picklefile.close()
                picklefile = open(SAVEDIR+name+"_new_p_progression", "rb")
                new_p_progression = pickle.load(picklefile)
                picklefile.close()
                picklefile = open(SAVEDIR+name+"_new_p_progression_LIN", "rb")
                new_p_progression_LIN = pickle.load(picklefile)
                picklefile.close()
                picklefile = open(SAVEDIR+name+"_new_p_progression_velo", "rb")
                new_p_progression_velo = pickle.load(picklefile)
                picklefile.close()
                print("Loaded subset progressions new_p")

                # Fold cumul
                fold_cumul_binary_progress_or_not = np.append(fold_cumul_binary_progress_or_not, binary_progress_or_not)
                fold_cumul_new_p_progression_velo = np.append(fold_cumul_new_p_progression_velo, new_p_progression_velo)
                fold_cumul_new_p_progression = np.append(fold_cumul_new_p_progression, new_p_progression)
                fold_cumul_new_p_progression_LIN = np.append(fold_cumul_new_p_progression_LIN, new_p_progression_LIN)

                # fpr, tpr
                fpr_velo, tpr_velo, threshold_auc_velo = metrics.roc_curve(binary_progress_or_not, new_p_progression_velo) #(y_test, preds)
                fpr_nlme, tpr_nlme, threshold_auc_nlme = metrics.roc_curve(binary_progress_or_not, new_p_progression) #(y_test, preds)
                fpr_LIN, tpr_LIN, threshold_auc_LIN = metrics.roc_curve(binary_progress_or_not, new_p_progression_LIN) #(y_test, preds)
                # precision, recall
                #print("binary_progress_or_not", binary_progress_or_not)
                #print("new_p_progression", new_p_progression)
                
                precision_velo, recall_velo, threshold_aupr_velo = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression_velo) #(y_test, preds)
                precision_nlme, recall_nlme, threshold_aupr_nlme = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression) #(y_test, preds)
                precision_LIN, recall_LIN, threshold_aupr_LIN = metrics.precision_recall_curve(binary_progress_or_not, new_p_progression_LIN) #(y_test, preds)

                # Stored
                stored_fpr_velo.append(fpr_velo)
                stored_tpr_velo.append(tpr_velo)
                stored_fpr_nlme.append(fpr_nlme)
                stored_tpr_nlme.append(tpr_nlme)
                stored_fpr_LIN.append(fpr_LIN)
                stored_tpr_LIN.append(tpr_LIN)
                stored_precision_velo.append(precision_velo)
                stored_recall_velo.append(recall_velo)
                stored_precision_nlme.append(precision_nlme)
                stored_recall_nlme.append(recall_nlme)
                stored_precision_LIN.append(precision_LIN)
                stored_recall_LIN.append(recall_LIN)

                assert len(tpr_velo) == len(threshold_auc_velo)
                # Best threshold is lowest threshold with TPR above 0.8
                #print("threshold_auc_velo:", threshold_auc_velo)
                #print("threshold_auc_nlme:", threshold_auc_nlme)
                #print("threshold_auc_LIN:", threshold_auc_LIN)

                #print("tpr_velo:", tpr_velo)
                #print("tpr_nlme:", tpr_nlme)
                #print("tpr_LIN:", tpr_LIN)

                #print("fpr_velo:", fpr_velo)
                #print("fpr_nlme:", fpr_nlme)
                #print("fpr_LIN:", fpr_LIN)

                if threshold_logic == "reach 80 % TPR": # 1
                    best_threshold_velo = next((threshold for tpr, threshold in zip(tpr_velo, threshold_auc_velo) if tpr >= 0.8), None)
                    best_threshold_nlme = next((threshold for tpr, threshold in zip(tpr_nlme, threshold_auc_nlme) if tpr >= 0.8), None)
                    best_threshold_LIN = next((threshold for tpr, threshold in zip(tpr_LIN, threshold_auc_LIN) if tpr >= 0.8), None)
                    #print("best_threshold_velo:", best_threshold_velo)
                    #print("best_threshold_nlme:", best_threshold_nlme)
                    #print("best_threshold_LIN:", best_threshold_LIN)
                elif threshold_logic == "maximize F1": # 2
                    #print("recall_nlme", recall_nlme)
                    #print("precision_nlme", precision_nlme)
                    #print("threshold_aupr_nlme", threshold_aupr_nlme)
                    f1_scores_velo = 2 * (precision_velo * recall_velo) / (recall_velo + precision_velo)
                    f1_scores_nlme = 2 * (precision_nlme * recall_nlme) / (recall_nlme + precision_nlme)
                    f1_scores_LIN = 2 * (precision_LIN * recall_LIN) / (recall_LIN + precision_LIN)
                    #print("f1_scores_velo", f1_scores_velo)
                    #print("f1_scores_nlme", f1_scores_nlme)
                    #print("f1_scores_LIN", f1_scores_LIN)
                    threshold_aupr_velo = np.append(threshold_aupr_velo, np.array([np.inf]))
                    threshold_aupr_nlme = np.append(threshold_aupr_nlme, np.array([np.inf]))
                    threshold_aupr_LIN = np.append(threshold_aupr_LIN, np.array([np.inf]))
                    #print("thresholds_f1_velo:", threshold_aupr_velo)
                    #print("thresholds_f1_nlme:", threshold_aupr_nlme)
                    #print("thresholds_f1_LIN:", threshold_aupr_LIN)
                    assert len(threshold_aupr_velo) == len(f1_scores_velo)
                    best_threshold_velo = threshold_aupr_velo[np.nanargmax(f1_scores_velo)]
                    best_threshold_nlme = threshold_aupr_nlme[np.nanargmax(f1_scores_nlme)]
                    best_threshold_LIN = threshold_aupr_LIN[np.nanargmax(f1_scores_LIN)]
                    #print("best_threshold_velo:", best_threshold_velo)
                    #print("best_threshold_nlme:", best_threshold_nlme)
                    #print("best_threshold_LIN:", best_threshold_LIN)
                    # Riktig! F eks 1 hvis p=0.4 eller stoerre                
                elif threshold_logic == "be "+"{:.2f}".format(predetermined_threshold)+" for NLME and "+"{:.2f}".format(predetermined_threshold_velo)+" for velo":
                    best_threshold_velo = predetermined_threshold_velo
                    best_threshold_nlme = predetermined_threshold
                    best_threshold_LIN = predetermined_threshold
                elif threshold_logic == "reach 80 % TPR for first 10, then fixed to fold 10 threshold": # 4
                    if CLIP_MPROTEIN_TIME > 1+10*28:
                        # Check if saved threshold from 10
                        assert min(velo_threshold_80_trp_from_clip_10, nlme_threshold_80_trp_from_clip_10, LIN_threshold_80_trp_from_clip_10) > 0, "Threshold for 10 has not been saved"
                        best_threshold_velo = velo_threshold_80_trp_from_clip_10
                        best_threshold_nlme = nlme_threshold_80_trp_from_clip_10
                        best_threshold_LIN = LIN_threshold_80_trp_from_clip_10
                    else:
                        # Find lowest threshold that puts TPR above 0.8
                        best_threshold_velo = next((threshold for tpr, threshold in zip(tpr_velo, threshold_auc_velo) if tpr >= 0.8), None)
                        best_threshold_nlme = next((threshold for tpr, threshold in zip(tpr_nlme, threshold_auc_nlme) if tpr >= 0.8), None)
                        best_threshold_LIN = next((threshold for tpr, threshold in zip(tpr_LIN, threshold_auc_LIN) if tpr >= 0.8), None)
                        if CLIP_MPROTEIN_TIME == 1+10*28:
                            velo_threshold_80_trp_from_clip_10 = best_threshold_velo
                            nlme_threshold_80_trp_from_clip_10 = best_threshold_nlme
                            LIN_threshold_80_trp_from_clip_10 = best_threshold_LIN
                elif threshold_logic == "reach 80 % TPR for first 5, then fixed to fold 5 threshold": # 5
                    if CLIP_MPROTEIN_TIME > 1+5*28:
                        # Check if saved threshold from 10
                        assert min(velo_threshold_80_trp_from_clip_10, nlme_threshold_80_trp_from_clip_10, LIN_threshold_80_trp_from_clip_10) > 0, "Threshold for 10 has not been saved"
                        best_threshold_velo = velo_threshold_80_trp_from_clip_10
                        best_threshold_nlme = nlme_threshold_80_trp_from_clip_10
                        best_threshold_LIN = LIN_threshold_80_trp_from_clip_10
                    else:
                        # Find lowest threshold that puts TPR above 0.8
                        best_threshold_velo = next((threshold for tpr, threshold in zip(tpr_velo, threshold_auc_velo) if tpr >= 0.8), None)
                        best_threshold_nlme = next((threshold for tpr, threshold in zip(tpr_nlme, threshold_auc_nlme) if tpr >= 0.8), None)
                        best_threshold_LIN = next((threshold for tpr, threshold in zip(tpr_LIN, threshold_auc_LIN) if tpr >= 0.8), None)
                        if CLIP_MPROTEIN_TIME == 1+5*28:
                            velo_threshold_80_trp_from_clip_10 = best_threshold_velo
                            nlme_threshold_80_trp_from_clip_10 = best_threshold_nlme
                            LIN_threshold_80_trp_from_clip_10 = best_threshold_LIN
                elif threshold_logic == "average of threshold that reaches 80 % TPR for 4 and 5 across folds": # 6
                    if CLIP_MPROTEIN_TIME < 1+6*28:
                        # Find lowest threshold that puts TPR above 0.8
                        best_threshold_velo = next((threshold for tpr, threshold in zip(tpr_velo, threshold_auc_velo) if tpr >= 0.8), None)
                        best_threshold_nlme = next((threshold for tpr, threshold in zip(tpr_nlme, threshold_auc_nlme) if tpr >= 0.8), None)
                        best_threshold_LIN = next((threshold for tpr, threshold in zip(tpr_LIN, threshold_auc_LIN) if tpr >= 0.8), None)
                        if CLIP_MPROTEIN_TIME == 1+4*28:
                            velo_threshold_80_trp_from_clip_4 = best_threshold_velo
                            nlme_threshold_80_trp_from_clip_4 = best_threshold_nlme
                            LIN_threshold_80_trp_from_clip_4 = best_threshold_LIN
                        elif CLIP_MPROTEIN_TIME == 1+5*28:
                            velo_threshold_80_trp_from_clip_4_5 = (best_threshold_velo + velo_threshold_80_trp_from_clip_4) / 2
                            nlme_threshold_80_trp_from_clip_4_5 = (best_threshold_nlme + nlme_threshold_80_trp_from_clip_4) / 2
                            LIN_threshold_80_trp_from_clip_4_5 = (best_threshold_LIN + LIN_threshold_80_trp_from_clip_4) / 2
                    else:
                        # Check if saved threshold from 10
                        assert min(velo_threshold_80_trp_from_clip_4_5, nlme_threshold_80_trp_from_clip_4_5, LIN_threshold_80_trp_from_clip_4_5) > 0, "Threshold for 10 has not been saved"
                        best_threshold_velo = velo_threshold_80_trp_from_clip_4_5
                        best_threshold_nlme = nlme_threshold_80_trp_from_clip_4_5
                        best_threshold_LIN = LIN_threshold_80_trp_from_clip_4_5
                elif threshold_logic == "average max F1 threshold from 1-5 from 6 and up": # 7
                    # Initialize the lists only once
                    if CLIP_MPROTEIN_TIME == 1+1*28:
                        list_of_thresholds_velo = []
                        list_of_thresholds_nlme = []
                        list_of_thresholds_LIN = []
                    # Then fill them
                    if CLIP_MPROTEIN_TIME < 1+6*28:
                        f1_scores_velo = 2 * (precision_velo * recall_velo) / (recall_velo + precision_velo)
                        f1_scores_nlme = 2 * (precision_nlme * recall_nlme) / (recall_nlme + precision_nlme)
                        f1_scores_LIN = 2 * (precision_LIN * recall_LIN) / (recall_LIN + precision_LIN)
                        threshold_aupr_velo = np.append(threshold_aupr_velo, np.array([np.inf]))
                        threshold_aupr_nlme = np.append(threshold_aupr_nlme, np.array([np.inf]))
                        threshold_aupr_LIN = np.append(threshold_aupr_LIN, np.array([np.inf]))
                        assert len(threshold_aupr_velo) == len(f1_scores_velo)
                        best_threshold_velo = threshold_aupr_velo[np.nanargmax(f1_scores_velo)]
                        best_threshold_nlme = threshold_aupr_nlme[np.nanargmax(f1_scores_nlme)]
                        best_threshold_LIN = threshold_aupr_LIN[np.nanargmax(f1_scores_LIN)]
                        list_of_thresholds_velo.append(best_threshold_velo)
                        list_of_thresholds_nlme.append(best_threshold_nlme)
                        list_of_thresholds_LIN.append(best_threshold_LIN)
                    # And use them
                    else:
                        #print("list_of_thresholds_velo", list_of_thresholds_velo)
                        #print("list_of_thresholds_nlme", list_of_thresholds_nlme)
                        #print("list_of_thresholds_LIN", list_of_thresholds_LIN)
                        best_threshold_velo = np.mean(list_of_thresholds_velo)
                        best_threshold_nlme = np.mean(list_of_thresholds_nlme)
                        best_threshold_LIN = np.mean(list_of_thresholds_LIN)
                elif threshold_logic == "": # 8
                    # Initialize the lists only once
                    if CLIP_MPROTEIN_TIME == 1+1*28:
                        list_of_thresholds_velo = []
                        list_of_thresholds_nlme = []
                        list_of_thresholds_LIN = []
                    # Then fill them
                    if CLIP_MPROTEIN_TIME < 1+6*28:
                        f1_scores_velo = 2 * (precision_velo * recall_velo) / (recall_velo + precision_velo)
                        f1_scores_nlme = 2 * (precision_nlme * recall_nlme) / (recall_nlme + precision_nlme)
                        f1_scores_LIN = 2 * (precision_LIN * recall_LIN) / (recall_LIN + precision_LIN)
                        threshold_aupr_velo = np.append(threshold_aupr_velo, np.array([np.inf]))
                        threshold_aupr_nlme = np.append(threshold_aupr_nlme, np.array([np.inf]))
                        threshold_aupr_LIN = np.append(threshold_aupr_LIN, np.array([np.inf]))
                        assert len(threshold_aupr_velo) == len(f1_scores_velo)
                        best_threshold_velo = threshold_aupr_velo[np.nanargmax(f1_scores_velo)]
                        best_threshold_nlme = threshold_aupr_nlme[np.nanargmax(f1_scores_nlme)]
                        best_threshold_LIN = threshold_aupr_LIN[np.nanargmax(f1_scores_LIN)]
                        list_of_thresholds_velo.append(best_threshold_velo)
                        list_of_thresholds_nlme.append(best_threshold_nlme)
                        list_of_thresholds_LIN.append(best_threshold_LIN)
                    # And use them
                    else:
                        #print("list_of_thresholds_velo", list_of_thresholds_velo)
                        #print("list_of_thresholds_nlme", list_of_thresholds_nlme)
                        #print("list_of_thresholds_LIN", list_of_thresholds_LIN)
                        best_threshold_velo = np.mean(list_of_thresholds_velo)
                        best_threshold_nlme = np.mean(list_of_thresholds_nlme)
                        best_threshold_LIN = np.mean(list_of_thresholds_LIN)

                best_prediction_velo = [1 if p >= best_threshold_velo else 0 for p in new_p_progression_velo]
                best_prediction_nlme = [1 if p >= best_threshold_nlme else 0 for p in new_p_progression]
                best_prediction_LIN = [1 if p >= best_threshold_LIN else 0 for p in new_p_progression_LIN]
                #print("best_prediction_velo:", best_prediction_velo)
                #print("best_prediction_nlme:", best_prediction_nlme)
                #print("best_prediction_LIN:", best_prediction_LIN)

                # F1 score
                best_f1_velo[clip_index, dummy_fold_index] = metrics.f1_score(binary_progress_or_not, best_prediction_velo)
                best_f1_nlme[clip_index, dummy_fold_index] = metrics.f1_score(binary_progress_or_not, best_prediction_nlme)
                best_f1_LIN[clip_index, dummy_fold_index] = metrics.f1_score(binary_progress_or_not, best_prediction_LIN)

                # Accuracy
                best_accuracy_velo[clip_index, dummy_fold_index] = metrics.accuracy_score(binary_progress_or_not, best_prediction_velo)
                best_accuracy_nlme[clip_index, dummy_fold_index] = metrics.accuracy_score(binary_progress_or_not, best_prediction_nlme)
                best_accuracy_LIN[clip_index, dummy_fold_index] = metrics.accuracy_score(binary_progress_or_not, best_prediction_LIN)
                #print("\nbest_accuracy_velo", best_accuracy_velo)
                #print("best_accuracy_nlme", best_accuracy_nlme)
                #print("best_accuracy_LIN", best_accuracy_LIN)

                # Precision
                best_precision_velo[clip_index, dummy_fold_index] = metrics.precision_score(binary_progress_or_not, best_prediction_velo)
                best_precision_nlme[clip_index, dummy_fold_index] = metrics.precision_score(binary_progress_or_not, best_prediction_nlme)
                best_precision_LIN[clip_index, dummy_fold_index] = metrics.precision_score(binary_progress_or_not, best_prediction_LIN)
                #print("\nbest_precision_velo", best_precision_velo)
                #print("best_precision_nlme", best_precision_nlme)
                #print("best_precision_LIN", best_precision_LIN)

                best_recall_velo[clip_index, dummy_fold_index] = metrics.recall_score(binary_progress_or_not, best_prediction_velo)
                best_recall_nlme[clip_index, dummy_fold_index] = metrics.recall_score(binary_progress_or_not, best_prediction_nlme)
                best_recall_LIN[clip_index, dummy_fold_index] = metrics.recall_score(binary_progress_or_not, best_prediction_LIN)
                #print("\nbest_recall_velo", best_recall_velo)
                #print("best_recall_nlme", best_recall_nlme)
                #print("best_recall_LIN", best_recall_LIN)

                tn_velo, fp_velo, fn_velo, tp_velo = metrics.confusion_matrix(binary_progress_or_not, best_prediction_velo).ravel()
                tn_nlme, fp_nlme, fn_nlme, tp_nlme = metrics.confusion_matrix(binary_progress_or_not, best_prediction_nlme).ravel()
                tn_LIN, fp_LIN, fn_LIN, tp_LIN = metrics.confusion_matrix(binary_progress_or_not, best_prediction_LIN).ravel()

                best_specificity_velo[clip_index, dummy_fold_index] = tn_velo / (tn_velo+fp_velo)
                best_specificity_nlme[clip_index, dummy_fold_index] = tn_nlme / (tn_nlme+fp_nlme)
                best_specificity_LIN[clip_index, dummy_fold_index] = tn_LIN / (tn_LIN+fp_LIN)
                #print("\nbest_specificity_velo", best_specificity_velo)
                #print("best_specificity_nlme", best_specificity_nlme)
                #print("best_specificity_LIN", best_specificity_LIN)

                # AUC
                auc_velo[clip_index, dummy_fold_index] = metrics.auc(fpr_velo, tpr_velo)
                auc_nlme[clip_index, dummy_fold_index] = metrics.auc(fpr_nlme, tpr_nlme)
                auc_LIN[clip_index, dummy_fold_index] = metrics.auc(fpr_LIN, tpr_LIN)

                # Best threshold
                all_best_threshold_velo[clip_index, dummy_fold_index] = best_threshold_velo
                all_best_threshold_nlme[clip_index, dummy_fold_index] = best_threshold_nlme
                all_best_threshold_LIN[clip_index, dummy_fold_index] = best_threshold_LIN
                
                del binary_progress_or_not
                del new_p_progression
                del new_p_progression_LIN
                del new_p_progression_velo
                #gc.collect()
        
        # Plot thresholds 
        fig, ax = plt.subplots()
        plt.title("Threshold selected to "+threshold_logic+" for NLME without covariates")
        ax.axhline(y=predetermined_threshold, color="k", linestyle="--")
        for dummy_fold_index, actual_fold_index in enumerate(which_folds):
            threshold_vals = all_best_threshold_nlme[:,dummy_fold_index]
            clip_values = range(1,len(threshold_vals)+1)
            plt.plot(clip_values, threshold_vals)
        plt.ylim(0,1)
        plt.ylabel("Threshold")
        plt.xlabel("Cycles")
        if t_choice == 1:
            plt.savefig(PLOTDIR+name+"_threshold_nlme_80tpr.pdf", dpi=300)
        elif t_choice == 2: 
            plt.savefig(PLOTDIR+name+"_threshold_nlme_maxf1.pdf", dpi=300)
        elif t_choice == 4:
            plt.savefig(PLOTDIR+name+"_threshold_nlme_80tpr_before_10.pdf", dpi=300)
        plt.close()

        # Plot thresholds 
        fig, ax = plt.subplots()
        plt.title("Threshold selected to "+threshold_logic+" for NLME with covariates")
        ax.axhline(y=predetermined_threshold, color="k", linestyle="--")
        for dummy_fold_index, actual_fold_index in enumerate(which_folds):
            threshold_vals = all_best_threshold_LIN[:,dummy_fold_index]
            clip_values = range(1,len(threshold_vals)+1)
            plt.plot(clip_values, threshold_vals)
        plt.ylim(0,1)
        plt.ylabel("Threshold")
        plt.xlabel("Cycles")
        if t_choice == 1:
            plt.savefig(PLOTDIR+name+"_threshold_LIN_80tpr.pdf", dpi=300)
        elif t_choice == 2: 
            plt.savefig(PLOTDIR+name+"_threshold_LIN_maxf1.pdf", dpi=300)
        elif t_choice == 4:
            plt.savefig(PLOTDIR+name+"_threshold_LIN_80tpr_before_10.pdf", dpi=300)
        plt.close()

        # Plot thresholds 
        fig, ax = plt.subplots()
        plt.title("Threshold selected to "+threshold_logic+" for velocity model")
        ax.axhline(y=predetermined_threshold_velo, color="k", linestyle="--")
        for dummy_fold_index, actual_fold_index in enumerate(which_folds):
            threshold_vals = all_best_threshold_velo[:,dummy_fold_index]
            clip_values = range(1,len(threshold_vals)+1)
            plt.plot(clip_values, threshold_vals)
        plt.ylim(0,1)
        plt.ylabel("Threshold")
        plt.xlabel("Cycles")
        if t_choice == 1:
            plt.savefig(PLOTDIR+name+"_threshold_velo_80tpr.pdf", dpi=300)
        elif t_choice == 2: 
            plt.savefig(PLOTDIR+name+"_threshold_velo_maxf1.pdf", dpi=300)
        elif t_choice == 4:
            plt.savefig(PLOTDIR+name+"_threshold_velo_80tpr_before_10.pdf", dpi=300)
        plt.close()
        
        model_names = ["Velocity model", "NLME", "NLME w/covariates"]
        color_array = [plt.cm.viridis(0.9), plt.cm.viridis(0.6), plt.cm.viridis(0.3)]
        bar_color_array = ["dimgrey", "dimgrey", "dimgrey"] #["dimgrey", "darkgrey", "darkgrey"] #, "lightgrey"]
        auc_all = [auc_velo, auc_nlme, auc_LIN]
        accuracy_all = [best_accuracy_velo, best_accuracy_nlme, best_accuracy_LIN]
        f1_all = [best_f1_velo, best_f1_nlme, best_f1_LIN]
        precision_all = [best_precision_velo, best_precision_nlme, best_precision_LIN]
        recall_all = [best_recall_velo, best_recall_nlme, best_recall_LIN]
        specificity_all = [best_specificity_velo, best_specificity_nlme, best_specificity_LIN]
        metric_names = ["AUC", "Accuracy", "F1 score", "Precision", "Sensitivity", "Specificity"]
        all_metrics = [auc_all, accuracy_all, f1_all, precision_all, recall_all, specificity_all]
        # Bar plot
        bar_width = 0.25
        # xticks
        #tick_positions = np.arange(len(pred_window_starts))
        #tick_labels = [int(np.rint( (cliptime - 1) // 28 )) for cliptime in pred_window_starts]
        step_size = 3
        latest_clip_id = int(np.rint( (pred_window_starts[-1] - 1) // 28 ))
        tick_labels = [1] + [cs for cs in range(step_size, latest_clip_id, step_size)]
        tick_positions = np.array(tick_labels) - 1
        """
        latest_cycle_start = int(np.rint( (pred_window_starts[-1] - 1) // 28 + 1 ))
        step_size = 3
        tick_labels = [1] + [cs for cs in range(step_size, latest_cycle_start, step_size)]
        if len(tick_labels) < 4:
            step_size = 1
            tick_labels = [1] + [cs for cs in range(1 + step_size, latest_cycle_start, step_size)] + [latest_cycle_start]
        if len(tick_labels) < 2:
            tick_labels = [1, latest_cycle_start]
        tick_positions = [(cycle_nr - 1) * 28 + 1 for cycle_nr in tick_labels]
        """
        fig, axs = plt.subplots(1,len(metric_names)+1, figsize=(20,5)) #, figsize=(16,5))
        fig.suptitle("Threshold selected to "+threshold_logic)
        for metric_index, _ in enumerate(metric_names):
            ax = axs[metric_index]
            ax.set_title(metric_names[metric_index])
            ax.set_ylim(0,1)
            # Plot all cliptimes together
            for model_index, modelname in enumerate(model_names):
                x_locations = np.arange(len(pred_window_starts)) - bar_width
                # Bar for mean value
                offset = model_index*bar_width
                fold_values = all_metrics[metric_index][model_index] # Has 2 dimensions left: Cliptimes, and splits (folds)
                print("fold_values", fold_values)
                meanvals = np.mean(fold_values, axis=-1)
                stdvals = np.std(fold_values, axis=-1)
                ## 95 % confidence intervals of the mean, assuming normally distributed means, which would be true only at larger sample sizes:
                ## Standard error of the mean
                #stderrofmean = stdvals / np.sqrt(n_splits)
                #ax.errorbar(x=x_locations+offset, y=meanvals, yerr=1.96*stderrofmean)

                # Confidence intervals of the mean, with t-distribution assumption
                degrees_of_freedom = n_splits - 1
                # Critical_value for 0.95 (two-sided 90 % CI):
                critval_table_90_CI = {2 : 2.920, 3 : 2.353, 4 : 2.132, 5 : 2.015, 6 : 1.943}
                # Critical_value for 0.975 (two-sided 95 % CI):
                critval_table_95_CI = {2 : 4.303, 3 : 3.182, 4 : 2.776, 5 : 2.571, 6 : 2.447}
                critical_value_with_correct_dof = critval_table_90_CI[degrees_of_freedom]
                # 2.93 with 15 or so? For a different level then. 
                # For each cliptime in pred_window_starts, calculate the plain sum of square errors between fold values and fold means! 
                # Difference with mean works because fold_values comes from a numpy array
                sum_of_square_error_values = np.array([sum((fold_values[cc,:] - meanvals[cc])**2) for cc, _ in enumerate(pred_window_starts)])
                rects = ax.bar(x=x_locations+offset, height=np.mean(fold_values, axis=-1), width=bar_width, color=color_array[model_index]) #color=color_array[model_index])
                #ax.errorbar(x=x_locations+offset, y=meanvals, yerr=critical_value_with_correct_dof*(np.sqrt(sum_of_square_error_values/degrees_of_freedom))/np.sqrt(n_splits), fmt=".", color=bar_color_array[model_index]) #, capsize=1, markersize=0, elinewidth=1) #fmt=".", color=color_array[model_index]) # markersize=5, linewidth=1
                
                # std
                #ax.errorbar(x=x_locations+offset, y=meanvals, yerr=stdvals, fmt=".", color=bar_color_array[model_index]) #, color="k") #, capsize=1, markersize=0, elinewidth=1) #fmt=".", color=color_array[model_index]) # markersize=5, linewidth=1

            ax.set_xticks(tick_positions, tick_labels)
            ax.set_xlabel("Observed cycles")
        axs[-1].axis("off")
        legend_handles = [mpatches.Rectangle((0,0), 1, 1, label=model_name, color=color) for model_name, color in zip(model_names, color_array)]
        plt.tight_layout()
        axs[-1].legend(handles=legend_handles, loc="center left") #, bbox_to_anchor=(0.1,0.1))
        crossfold_name = dataset+"_"+df_arm+"_"+likelihood_model+"_using_"+str(CLIP_MPROTEIN_TIME)+"_days_predict_"+str(pred_window_length)+"_ahead_"+str(N_patients)+"_patients_"+str(N_progressors)+"_progressors"
        if t_choice == 1:
            plt.savefig(PLOTDIR+crossfold_name+"_bars_80tpr.pdf", dpi=300)
        elif t_choice == 2: 
            plt.savefig(PLOTDIR+crossfold_name+"_bars_maxf1.pdf", dpi=300)
        elif t_choice == 3: 
            plt.savefig(PLOTDIR+crossfold_name+"_bars_fixed_threshold.pdf", dpi=300)
        elif t_choice == 4:
            plt.savefig(PLOTDIR+crossfold_name+"_bars_80tpr_before_10.pdf", dpi=300)
        #plt.show()
        # Save again with each fold as black mark
        for metric_index, _ in enumerate(metric_names):
            ax = axs[metric_index]
            # Plot all cliptimes together
            for model_index, modelname in enumerate(model_names):
                x_locations = np.arange(len(pred_window_starts)) - bar_width
                # Bar for mean value
                offset = model_index*bar_width
                fold_values = all_metrics[metric_index][model_index]
                # Separate for each fold
                for dummy_fold_index, actual_fold_index in enumerate(which_folds):
                    ax.plot(x_locations + offset, fold_values[:, dummy_fold_index], color="k", marker="2", markersize=5, linestyle="") #, markeredgewidth=2, markeredgecolor="k")
        if t_choice == 1:
            plt.savefig(PLOTDIR+crossfold_name+"_bars_80tpr_singlefolds.pdf", dpi=300)
        elif t_choice == 2: 
            plt.savefig(PLOTDIR+crossfold_name+"_bars_maxf1_singlefolds.pdf", dpi=300)    
        elif t_choice == 3: 
            plt.savefig(PLOTDIR+crossfold_name+"_bars_fixed_threshold_singlefolds.pdf", dpi=300)    
        elif t_choice == 4:
            plt.savefig(PLOTDIR+crossfold_name+"_bars_80tpr_before_10_singlefolds.pdf", dpi=300)
        plt.close()

##########################################
#### control panel #######################
##########################################
if __name__ == "__main__":
    # 1 load data
    # The following was the workflow on Vivli server
    """
    # Workdir is evenmmV
    # Load patient dictionaries for both arms
    with open('./binaries_and_pickles/patient_dict_IKEMA_Kd', 'rb') as picklefile:
        patient_dict_Kd = pickle.load(picklefile)
    with open('./binaries_and_pickles/patient_dict_IKEMA_IKd', 'rb') as picklefile:
        patient_dict_IKd = pickle.load(picklefile)
    # All patients
    patient_dict_all = deepcopy(patient_dict_Kd)
    for patient_name, patient in patient_dict_IKd.items():
        patient_dict_all[patient_name] = patient
    N_patients_all, t_all, Y_all = get_N_t_Y(patient_dict_all)
    print("Creating X for All patients:")
    X_all, patient_dictionary_complete_all, raw_floats_adtte, raw_floats_adsl = create_X_IKEMA_folds(patient_dict_all)
    """
    plot_cumulative() #X_all, patient_dictionary_complete_all)
