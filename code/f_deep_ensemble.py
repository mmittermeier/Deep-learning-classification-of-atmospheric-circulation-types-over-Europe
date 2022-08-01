# -*- coding: utf-8 -*-
"""

Postprocessing of neural network results

copyright: Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich
citation: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig,
    R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types
    over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental
    Research Letters. 17. doi: 10.1088/1748-9326/ac8068.
@author: m.mittermeier@lmu.de, M. Weigert
"""


import os
import sys
import numpy as np
import pandas as pd


# Function to read all files of a specific type:
def read_results_files(path_results, period, years, member, name_dataset, run_family, network_version, num_inits=30):

    # Search folders for data to read:
    ensemble = []
    for i in range(0, num_inits):
        x = np.load(path_results + os.sep + "1_final-model_ppc_"+network_version+str(i+1) + os.sep + run_family)
        ensemble.append(x)
    ensemble = np.stack(ensemble)
    print("got ensemble")
    return(ensemble)


# Function to read all files of a specific type:
def read_confusion_files(path_confusion, class_names, num_inits=30):

    # Search folders for data to read:
    fscores = []
    for i in range(0, num_inits):
        path_file = path_confusion + os.sep + "1_final-model_ppc_v6-2-init" + str(i + 1) + os.sep + "confusion_train_final_model_v6-2-init" + str(i+1) + ".npy"
        x = np.load(path_file)
        confusion_subset = pd.DataFrame(x)
        accuracy = np.trace(confusion_subset) / np.sum(np.concatenate(confusion_subset)) * 100
        sum_labels_classes = np.sum(confusion_subset, axis=0)
        sum_pred_classes = np.sum(confusion_subset, axis=1)
        precision_classes = np.ones(len(class_names)) * np.nan
        recall_classes = np.ones(len(class_names)) * np.nan
        fscore_classes = np.ones(len(class_names)) * np.nan
        for class_name in enumerate(confusion_subset):
            c = class_name[0]
            precision_classes[c] = (confusion_subset[class_name[1]][c]/sum_pred_classes[c])*100
            recall_classes[c] = (confusion_subset[class_name[1]][c]/sum_labels_classes[c])*100
            fscore_classes[c] = 2*((precision_classes[c]*recall_classes[c])/(precision_classes[c]+recall_classes[c]+sys.float_info.epsilon))
        fscores.append(fscore_classes)
    fscores = np.stack(fscores)    
    return(fscores)


def apply_three_day_rule_ensemble(ensemble, pred_class_in):
    pred_class = pred_class_in.copy()
    pred_prob = pd.DataFrame(ensemble)

    # Step 1 : Correction of one-day-classifications:
    for i in range(len(pred_class) - 1):
        if pred_class[i] != pred_class[i - 1] and pred_class[i] != pred_class[i + 1]:
            # Same class before and after current time point:
            if pred_class[i - 1] == pred_class[i + 1]:  # neighborhood consistency
                pred_class[i] = pred_class[i + 1]
            elif i == 0:
                pred_class[i] = pred_class[i + 1]  # first value
            elif i == len(pred_class):
                pred_class[i] = pred_class[i - 1]  #

            # Different classes before and after:
            elif pred_class[i - 1] != pred_class[i + 1]:

                # Search for the last and the next class lasting at least three days:
                if pred_prob.loc[i, int(pred_class[i - 1])] > pred_prob.loc[i, int(pred_class[i + 1])]:
                    pred_class[i] = pred_class[i - 1]
                else:
                    pred_class[i] = pred_class[i + 1]

    # Step 2: Correction of two-day-classifications:
    for i in range(len(pred_class) - 2):
        if pred_class[i] != pred_class[i - 1] and pred_class[i] != pred_class[i + 2]:
            # Same class before and after current time point:
            if pred_class[i - 1] == pred_class[i + 2]:
                pred_class[i] = pred_class[i + 2]
                pred_class[i + 1] = pred_class[i + 2]

            # Different classes before and after:
            elif pred_class[i - 1] != pred_class[i + 2]:

                # Search for the last and the next class lasting at least three days:
                if pred_prob.loc[i, int(pred_class[i - 1])] > pred_prob.loc[i, int(pred_class[i + 2])]:
                    pred_class[i] = pred_class[i - 1]
                    if pred_prob.loc[i + 1, int(pred_class[i - 1])] > pred_prob.loc[i + 1, int(pred_class[i + 2])]:
                        pred_class[i + 1] = pred_class[i - 1]
                    else:
                        pred_class[i + 1] = pred_class[i + 2]
                else:
                    pred_class[i] = pred_class[i + 2]

    return pred_class


# Function to calculate ensemble mean:
def calculate_ensemble_mean(model, type, years, num_inits=30):

    # Read all files:
    ensemble = read_results_files(model=model, type=type, years=years, num_inits=num_inits)
    # Calculate mean per time point and GWL:
    mean_ensemble = np.mean(ensemble, axis=0)

    # Classification into class of highest value:
    y_pred_class = np.empty((mean_ensemble.shape[0]))
    for r in np.arange(0, mean_ensemble.shape[0]):
        y_pred_class[r] = np.argmax(mean_ensemble[r, :])

    # Application of three day rule:
    y_pred_class = apply_three_day_rule_ensemble(mean_ensemble, y_pred_class)

    return(mean_ensemble, y_pred_class)


def deep_ensemble_weighted_mean(ensemble, fscores):
    
    f_weights = fscores / np.sum(fscores, axis=0)
   
    # Calculate weighted mean per time point and GWL based on f-score:
    ensemble_r = np.transpose(ensemble, (1, 0, 2))
    weighted = ensemble_r * f_weights
    mean_ensemble = np.sum(weighted, axis=1)

    # Classification into class of highest value:
    y_pred_class = np.empty((mean_ensemble.shape[0]))
    for r in np.arange(0, mean_ensemble.shape[0]):
        y_pred_class[r] = np.argmax(mean_ensemble[r, :])

    # Application of three day rule:
    y_pred_class = apply_three_day_rule_ensemble(mean_ensemble, y_pred_class)
    
    return y_pred_class


# Function to run ensemble mean:
def run_ensemble_weighted_mean(path_results, path_confusion, path_out, name_dataset, period, years, member, network_version, class_names, num_inits=30):
    # Loop over runs/families to load data and calculate y_pred_class
    for run_family in [filename for filename in os.listdir(os.path.join(path_results, "1_final-model_ppc_"+network_version+"1")) if filename.startswith("Probability")]:
        print("run_family: ", run_family)
        filename_out = "Inference_deep-ensemble-mean" + run_family[11:]
        outpathdir = os.path.join(path_out + os.sep + "deep_ensemble_mean")
        if not os.path.exists(outpathdir):
            os.makedirs(outpathdir)
            
        # Read all files:
        ensemble = read_results_files(path_results, period, years, member, name_dataset, run_family, network_version, num_inits)
        print("get fscores")
        fscores = read_confusion_files(path_confusion, class_names, num_inits)

        y_pred_class = deep_ensemble_weighted_mean(ensemble, fscores)
        years = run_family[-28:-11]
        np.save(outpathdir + os.sep + filename_out, y_pred_class)
