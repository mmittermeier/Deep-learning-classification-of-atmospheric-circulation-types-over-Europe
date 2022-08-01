#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation functions to evaluate model 

- confusion matrices
- learning rates
- f-score

copyright: Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich
citation: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig,
    R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types
    over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental
    Research Letters. 17. doi: 10.1088/1748-9326/ac8068.
@author: m.mittermeier@lmu.de, M. Weigert
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import sys


def classify_predictions(X, model):
    # classification into class of highest value
    y_pred = model.predict(X)
    y_pred_class = y_pred[:, 0]
    for r in np.arange(0, X.shape[0]):
        y_pred_class[r] = np.argmax(y_pred[r, :])
    return y_pred_class


def apply_three_day_rule(X, model, pred_class_in):
    
    pred_class = pred_class_in.copy()
    pred_prob = pd.DataFrame(model.predict(X))

    # Step 1 : Transition smoothing: Correction of one-day classifications
    for i in range(len(pred_class) - 1):
        if pred_class[i] != pred_class[i - 1] and pred_class[i] != pred_class[i + 1]:
            # Same class before and after current time point:
            if pred_class[i - 1] == pred_class[i + 1]:   # neighborhood consistency
                pred_class[i] = pred_class[i + 1]
            elif i == 0:
                pred_class[i] = pred_class[i + 1]     # first value
            elif i == len(pred_class):
                pred_class[i] = pred_class[i - 1]     #

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


def summary_metrics(y_pred_decoded, y_obs_decoded, y_encoded_shape):
    confusion = np.rot90(confusion_matrix(y_obs_decoded, y_pred_decoded).T, 2) # structure with targets on x-axis and predictions on y-axis
    TP = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP/(TP+FP)
    recall = TP/(TP + FN)
    f_score = 2/((1/precision) + (1/recall))
    return confusion, precision, recall, f_score


def summary_metrics_confusion(confusion):
    TP = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP + FN)
    f_score = 2/((1/precision) + (1/recall))
    return precision, recall, f_score


def plot_learning_curves(history, path_fig, perform_metrics):

    "plot history of loss and accuracy"
    
    # Learning rate
    fig1 = plt.figure()
    h_loss = history.history['loss'] # history loss
    val_loss = history.history['val_loss']
    epochs = range(1, len(h_loss) + 1)
    plt.plot(epochs, h_loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_fig + os.sep + "learning_rate_loss.png")
    plt.close(fig1)
    
    # Learning rate for performance-metrics
    for p, perform_metric in enumerate(perform_metrics):
        fig2 = plt.figure()
        h_acc = history.history[perform_metric]
        val_acc = history.history["val_" + perform_metric]
        plt.plot(epochs, h_acc, 'y', label='Training ' + perform_metric)
        plt.plot(epochs, val_acc, 'r', label='Validation ' + perform_metric)
        plt.title('Training and validation ' + perform_metric)
        plt.xlabel('Epochs')
        plt.ylabel(perform_metric)
        plt.legend()
        plt.savefig(path_fig + os.sep + "learning_rate_" + perform_metric + ".png")
        plt.close(fig2)


def calculate_confusion_matrix_binary(filename, subset_name, y_pred_class_subset, trainingset, path_txt, perform_metrics, subset_perform1, return_f_score):
    
    if subset_name == "test":
        confusion_subset, precision_subset, recall_subset, f_score_subset = summary_metrics(y_pred_class_subset, trainingset.y_test, trainingset.y_test_encoded.shape[0])
    elif subset_name == "dev":
        confusion_subset, precision_subset, recall_subset, f_score_subset = summary_metrics(y_pred_class_subset, trainingset.y_dev, trainingset.y_dev_encoded.shape[0])
    elif subset_name == "train":
        confusion_subset, precision_subset, recall_subset, f_score_subset = summary_metrics(y_pred_class_subset, trainingset.y_train, trainingset.y_train_encoded.shape[0])
    else:
        raise ValueError("define subset")
    
    text_file = open(path_txt + os.sep + "DNN_metrics_" + subset_name + ".txt", "w")
    text_file.write("DNN Metrics: \n \n")
    text_file.write(subset_name + ' precision: {}'.format(np.round(precision_subset * 100, 2)) + '\n')
    text_file.write(subset_name + ' recall: {}'.format(np.round(recall_subset* 100, 2)) + '\n')
    text_file.write(subset_name + ' f-score: {}'.format(np.round(f_score_subset, 2)) + '\n')
    text_file.write(subset_name + " " + perform_metrics[0]+ ": {}".format(np.round(subset_perform1 * 100, 2)) + "\n")
    text_file.write('Confusion matrix (labels (1,0)): \n predictions(1) {}'.format(confusion_subset[0, :]) + '\n' + ' predictions(0) {}'.format(confusion_subset[1, :]))
    text_file.close()

    # save confusion
    np.save(path_txt + os.sep + subset_name + "_confusion_" + filename + ".npy", confusion_subset)
    
    if return_f_score == True and subset_name == "dev":
        return confusion_subset, f_score_subset
    else:
        return confusion_subset


def calculate_confusion_matrix_multiclass(filename, subset_name, y_pred_class_subset, trainingset, model, class_names, path_txt):
        
    # Calculation of confusion matrices:
    if subset_name == "test":
        confusion_subset = confusion_matrix(trainingset.y_test, y_pred_class_subset).T # structure with targets on x-axis and predictions on y-axis
        pred_subset = pd.DataFrame(model.predict(trainingset.x_test))
        pred_subset.columns = class_names.values()
        pred_subset["True_GWL"] = pd.Series(trainingset.y_test_label)
        pred_subset["Predicted_GWL"] = pd.Series(y_pred_class_subset)
        pred_subset["Predicted_GWL"] = pred_subset.Predicted_GWL.replace(class_names)
        dates_test = pd.DataFrame(trainingset.dates_test, columns = ("Year", "Month", "Day", "Hour"))
        dates_test['Date'] = dates_test.Year.astype(str).str.cat(dates_test.Month.astype(str), sep = '-')
        dates_test['Date'] = dates_test.Date.astype(str).str.cat(dates_test.Day.astype(str), sep = '-')
        pred_subset["Date"] = dates_test["Date"]
    elif subset_name == "dev":
        confusion_subset = confusion_matrix(trainingset.y_dev, y_pred_class_subset).T # structure with targets on x-axis and predictions on y-axis
        pred_subset = pd.DataFrame(model.predict(trainingset.x_dev))
        pred_subset.columns = class_names.values()
        pred_subset["True_GWL"] = pd.Series(trainingset.y_dev_label)
        pred_subset["Predicted_GWL"] = pd.Series(y_pred_class_subset)
        pred_subset["Predicted_GWL"] = pred_subset.Predicted_GWL.replace(class_names)
        dates_dev = pd.DataFrame(trainingset.dates_dev, columns = ("Year", "Month", "Day", "Hour"))
        dates_dev['Date'] = dates_dev.Year.astype(str).str.cat(dates_dev.Month.astype(str), sep = '-')
        dates_dev['Date'] = dates_dev.Date.astype(str).str.cat(dates_dev.Day.astype(str), sep = '-')
        pred_subset["Date"] = dates_dev["Date"]
    elif subset_name == "train":
        confusion_subset = confusion_matrix(trainingset.y_train, y_pred_class_subset).T # structure with targets on x-axis and predictions on y-axis
        pred_subset = pd.DataFrame(model.predict(trainingset.x_train))
        pred_subset.columns = class_names.values()
        pred_subset["True_GWL"] = pd.Series(trainingset.y_train_label)
        pred_subset["Predicted_GWL"] = pd.Series(y_pred_class_subset)
        pred_subset["Predicted_GWL"] = pred_subset.Predicted_GWL.replace(class_names)
        dates_train = pd.DataFrame(trainingset.dates_train, columns = ("Year", "Month", "Day", "Hour"))
        dates_train['Date'] = dates_train.Year.astype(str).str.cat(dates_train.Month.astype(str), sep = '-')
        dates_train['Date'] = dates_train.Date.astype(str).str.cat(dates_train.Day.astype(str), sep = '-')
        pred_subset["Date"] = dates_train["Date"]
    else:
        raise ValueError("define subset")

    # Save prediction probabilities to excel file
    pred_subset.to_excel(path_txt + os.sep + "DNN_probs_" + subset_name + "_" + filename + ".xlsx", index = False)
   
    # save confusion matrix to excel file    
    confusion_subset_excel = pd.DataFrame(confusion_subset)
    confusion_subset_excel.columns = class_names
    confusion_subset_excel.index = class_names
    confusion_subset_excel.to_excel(path_txt + os.sep + "DNN_confusion_" + subset_name + "_" + filename + ".xlsx")
    
    return confusion_subset


def conf_matrix_to_perform_metrics(confusion_subset, class_names, path_txt, file_name, subset_name, perform_metrics,
                                   subset_loss, subset_perform1, subset_perform2=[]):
    
    confusion_subset = pd.DataFrame(confusion_subset)
    accuracy = np.trace(confusion_subset) / np.sum(np.concatenate(confusion_subset)) * 100
    sum_labels_classes = np.sum(confusion_subset, axis = 0)
    sum_pred_classes = np.sum(confusion_subset, axis = 1)
    precision_classes = np.ones(len(class_names)) * np.nan
    recall_classes =  np.ones(len(class_names)) * np.nan
    fscore_classes = np.ones(len(class_names)) * np.nan
    for class_name in enumerate(confusion_subset):
        c = class_name[0]
        precision_classes[c] = (confusion_subset[class_name[1]][c]/sum_pred_classes[c])*100
        recall_classes[c] = (confusion_subset[class_name[1]][c]/sum_labels_classes[c])*100
        fscore_classes[c] = 2*((precision_classes[c]*recall_classes[c])/(precision_classes[c]+recall_classes[c]+sys.float_info.epsilon))
    macro_fscore = np.nanmean(fscore_classes)
    avg_precision = np.nanmean(precision_classes)
    avg_recall = np.nanmean(recall_classes)    
        
    # write .txt-file before 3-day-rule
    text_file = open(path_txt + os.sep + file_name + subset_name + ".txt", "w")
    text_file.write("DNN Evaluation: \n \n")
    if len(perform_metrics) == 1:
        text_file.write(subset_name + ' loss: {}'.format(np.round(subset_loss, 2)) + '\n')
        text_file.write(subset_name + " " + perform_metrics[0]+ ": {}".format(np.round(subset_perform1 * 100, 2)) + "\n")
    elif len(perform_metrics) == 2:
        text_file.write(subset_name + ' loss: {}'.format(np.round(subset_loss, 2)) + '\n')
        text_file.write(subset_name + " " + perform_metrics[1]+ ": {}".format(np.round(subset_perform2 * 100, 2)) + "\n")
        text_file.write(subset_name + " " + perform_metrics[0]+ ": {}".format(np.round(subset_perform1 * 100, 2)) + "\n")

    text_file.write(subset_name + ' average precision: {}'.format(np.round(avg_precision, 2)) + '\n')
    text_file.write(subset_name + ' average recall: {}'.format(np.round(avg_recall, 2)) + '\n')
    text_file.write(subset_name + ' macro f-score: {}'.format(np.round(macro_fscore, 2)) + '\n')
    text_file.close()
    
    return macro_fscore


def mean_confusion_cv(path_out, cv_option, n_fold, n_models, gwl_selection, class_names):
    
    # load confusion matrices and store in list
    conf_train = []
    conf_dev = []
    conf_test = []
    if cv_option == "cv_kfold_lockbox_yearly":
        
        for i in np.arange(1, n_fold + 1):
            for j in np.arange(0, n_models):
                confusion_train = np.load(path_out + os.sep + "fold_" + str(i) + os.sep + "model" + str(j) + os.sep + "train_confusion.npy", allow_pickle=True)
                conf_train.append(confusion_train)
                confusion_dev = np.load(path_out + os.sep + "fold_" + str(i) + os.sep + "model" + str(j) + os.sep + "dev_confusion.npy", allow_pickle=True)
                conf_dev.append(confusion_dev)
                confusion_test = np.load(path_out + os.sep + "fold_" + str(i) + os.sep + "model" + str(j) + os.sep + "test_confusion.npy", allow_pickle=True)
                conf_test.append(confusion_test)              
    else:
        for i in n_fold:
            for j in np.arange(0, n_models):
                confusion_train = np.load(path_out + os.sep + "model_seed" + str(i) + os.sep + "model" + str(j) + os.sep + "train_confusion.npy", allow_pickle=True)
                conf_train.append(confusion_train)
                confusion_dev = np.load(path_out + os.sep + "model_seed" + str(i) + os.sep + "model" + str(j) + os.sep + "dev_confusion.npy", allow_pickle=True)
                conf_dev.append(confusion_dev)
                confusion_test = np.load(path_out + os.sep + "model_seed" + str(i) + os.sep + "model" + str(j) + os.sep + "test_confusion.npy", allow_pickle=True)
                conf_test.append(confusion_test)
        
    # Mean confusion matrices
    mean_conf_train = np.mean(conf_train, axis=0)
    mean_conf_dev = np.mean(conf_dev, axis=0)
    mean_conf_test = np.mean(conf_test, axis=0)
    
    if gwl_selection == "binary":

        # Mean precision, recall and f-score
        precision_train, recall_train, f_score_train = summary_metrics_confusion(mean_conf_train)
        precision_dev, recall_dev, f_score_dev = summary_metrics_confusion(mean_conf_dev)
        precision_test, recall_test, f_score_test = summary_metrics_confusion(mean_conf_test)
           
        # Save results to .txt file
        text_file = open(path_out + os.sep + "cv_metrics_results.txt", "w")
        text_file.write("Cross validation results: \n \n" )
        text_file.write('Train precision: {}'.format(np.round(precision_train * 100, 2)) + '\n')
        text_file.write('Train recall: {}'.format(np.round(recall_train * 100, 2)) + '\n')
        text_file.write('Train f-score: {}'.format(np.round(f_score_train, 2)) + '\n')
        text_file.write('Train confusion matrix (labels (1,0)): \n predictions(1) {}   {}'.format(np.round(mean_conf_train[0][0],2), np.round(mean_conf_train[0][1],2)) + '\n' + ' predictions(0) {}   {}'.format(np.round(mean_conf_train[1][0],2), np.round(mean_conf_train[1][1],2)))
        text_file.write( "\n ------------------------------------------------ \n")
        text_file.write('Dev precision: {}'.format(np.round(precision_dev * 100, 2)) + '\n')
        text_file.write('Dev recall: {}'.format(np.round(recall_dev * 100, 2)) + '\n')
        text_file.write('Dev f-score: {}'.format(np.round(f_score_dev, 2)) + '\n')
        text_file.write('Dev confusion matrix (labels (1,0)): \n predictions(1) {}   {}'.format(np.round(mean_conf_dev[0][0],2), np.round(mean_conf_dev[0][1],2)) + '\n' + ' predictions(0) {}   {}'.format(np.round(mean_conf_dev[1][0],2), np.round(mean_conf_dev[1][1],2)))
        text_file.write( "\n ------------------------------------------------ \n")
        text_file.write('Test precision: {}'.format(np.round(precision_test * 100, 2)) + '\n')
        text_file.write('Test recall: {}'.format(np.round(recall_test * 100, 2)) + '\n')
        text_file.write('Test f-score: {}'.format(np.round(f_score_test, 2)) + '\n')
        text_file.write('Test confusion matrix (labels (1,0)): \n predictions(1) {}   {}'.format(np.round(mean_conf_test[0][0],2), np.round(mean_conf_test[0][1],2)) + '\n' + ' predictions(0) {}   {}'.format(np.round(mean_conf_test[1][0],2), np.round(mean_conf_test[1][1],2)))
        text_file.close()
        
    else:
        
        # Calculate accuracy from confusion matrix using an identity matrix
        i_matrix = np.identity(len(class_names)) 
        correct_pred_test = np.sum(mean_conf_test * i_matrix)
        false_pred_test = np.sum(mean_conf_test- (mean_conf_test * i_matrix))
        acc_test = correct_pred_test / (correct_pred_test + false_pred_test)
        
        correct_pred_dev = np.sum(mean_conf_dev * i_matrix)
        false_pred_dev = np.sum(mean_conf_dev - (mean_conf_dev * i_matrix))
        acc_dev = correct_pred_dev / (correct_pred_dev + false_pred_dev)
        
        correct_pred_train = np.sum(mean_conf_train * i_matrix)
        false_pred_train = np.sum(mean_conf_train - (mean_conf_train * i_matrix))
        acc_train = correct_pred_train / (correct_pred_train + false_pred_train)
        
        text_file = open(path_out + os.sep + "cv_metrics_results.txt", "w")
        text_file.write("Cross validation results: \n \n" )
        text_file.write('Train Accuracy: {}'.format(np.round(acc_train * 100, 2)) + '\n')
        text_file.write('Dev Accuracy: {}'.format(np.round(acc_dev * 100, 2)) + '\n')
        text_file.write('Test Accuracy: {}'.format(np.round(acc_test * 100, 2)) + '\n')
        text_file.close()
        
        # Save confusion matrix to excel file
        confusion_test = pd.DataFrame(mean_conf_test)
        confusion_test.columns = class_names.values()
        confusion_test.index = class_names
        confusion_test.to_excel(path_out + os.sep + "DNN_confusion_test.xlsx")
        confusion_dev = pd.DataFrame(mean_conf_dev)
        confusion_dev.columns = class_names.values()
        confusion_dev.index = class_names
        confusion_dev.to_excel(path_out + os.sep + "DNN_confusion_dev.xlsx")
        confusion_train = pd.DataFrame(mean_conf_train)
        confusion_train.columns = class_names.values()
        confusion_train.index = class_names
        confusion_train.to_excel(path_out + os.sep + "DNN_confusion_train.xlsx")


def mean_confusion_nested_cv(name_of_step, path_conf, path_out, n_models, gwl_selection, class_names, train_name, test_name, train_name_out = "mean_confusion_train", test_name_out = "mean_confusion_test",  best_fold = "none"):
    
    # load confusion matrices and store in list
    conf_train = []
    conf_test = []
        
    for i in np.arange(1, n_models + 1):

        confusion_train = np.load(path_conf  + str(i) + os.sep +  train_name + name_of_step + ".npy", allow_pickle=True)
        conf_train.append(confusion_train)
        confusion_test = np.load(path_conf + str(i) + os.sep + test_name + name_of_step + ".npy", allow_pickle=True)
        conf_test.append(confusion_test)              
        
    # mean confusion matrices
    mean_conf_train = np.mean(conf_train, axis=0)
    mean_conf_test = np.mean(conf_test, axis=0)
    
    if gwl_selection == "binary":

        # mean precision, recall and f-score
        precision_train, recall_train, f_score_train = summary_metrics_confusion(mean_conf_train)
        precision_test, recall_test, f_score_test = summary_metrics_confusion(mean_conf_test)
           
        # save results to .txt file
        text_file = open(path_out + os.sep + "cv_metrics_results.txt", "w")
        text_file.write("Cross validation results: \n \n" )
        if best_fold != "none":
            text_file.write("Using best parameters of fold {}" .format(best_fold + 1) + "\n \n" )
        text_file.write('Train precision: {}'.format(np.round(precision_train * 100, 2)) + '\n')
        text_file.write('Train recall: {}'.format(np.round(recall_train * 100, 2)) + '\n')
        text_file.write('Train f-score: {}'.format(np.round(f_score_train, 2)) + '\n')
        text_file.write('Train confusion matrix (labels (1,0)): \n predictions(1) {}   {}'.format(np.round(mean_conf_train[0][0],2), np.round(mean_conf_train[0][1],2)) + '\n' + ' predictions(0) {}   {}'.format(np.round(mean_conf_train[1][0],2), np.round(mean_conf_train[1][1],2)))
        text_file.write( "\n ------------------------------------------------ \n")
        text_file.write('Test precision: {}'.format(np.round(precision_test * 100, 2)) + '\n')
        text_file.write('Test recall: {}'.format(np.round(recall_test * 100, 2)) + '\n')
        text_file.write('Test f-score: {}'.format(np.round(f_score_test, 2)) + '\n')
        text_file.write('Test confusion matrix (labels (1,0)): \n predictions(1) {}   {}'.format(np.round(mean_conf_test[0][0],2), np.round(mean_conf_test[0][1],2)) + '\n' + ' predictions(0) {}   {}'.format(np.round(mean_conf_test[1][0],2), np.round(mean_conf_test[1][1],2)))
        text_file.close()
        
        # save confusion matrices to .npy file
        np.save(path_out + os.sep + train_name_out + ".npy", mean_conf_train)
        np.save(path_out + os.sep + test_name_out + ".npy", mean_conf_test)
        
    else:
        
        # calculate accuracy from confusion matrix using an identity matrix
        i_matrix = np.identity(len(class_names)) 
        
        correct_pred_train = np.sum(mean_conf_train * i_matrix)
        false_pred_train = np.sum(mean_conf_train - (mean_conf_train * i_matrix))
        acc_train = correct_pred_train / (correct_pred_train + false_pred_train)
        
        correct_pred_test = np.sum(mean_conf_test * i_matrix)
        false_pred_test = np.sum(mean_conf_test- (mean_conf_test * i_matrix))
        acc_test = correct_pred_test / (correct_pred_test + false_pred_test)
        
        # F-scores
        sum_labels_classes = np.sum(mean_conf_test, axis = 0)
        sum_pred_classes = np.sum(mean_conf_test, axis = 1)
        precision_classes = np.ones(len(class_names)) * np.nan
        recall_classes =  np.ones(len(class_names)) * np.nan
        fscore_classes = np.ones(len(class_names)) * np.nan
        for class_nr in np.arange(0, len(class_names)):
            precision_classes[class_nr] = (mean_conf_test[class_nr, class_nr]/sum_pred_classes[class_nr])*100
            recall_classes[class_nr] = (mean_conf_test[class_nr, class_nr]/sum_labels_classes[class_nr])*100
            fscore_classes[class_nr] = 2*((precision_classes[class_nr]*recall_classes[class_nr])/(precision_classes[class_nr]+recall_classes[class_nr]+sys.float_info.epsilon))
        macro_fscore = np.nanmean(fscore_classes)
        avg_precision = np.nanmean(precision_classes)
        avg_recall = np.nanmean(recall_classes)
        
        text_file = open(path_out + os.sep + "cv_metrics_results_" + name_of_step + ".txt", "w")
        text_file.write("Cross validation results (train & test): \n \n" )
        text_file.write('Train Accuracy: {}'.format(np.round(acc_train * 100, 2)) + '\n')
        text_file.write('Test Accuracy: {}'.format(np.round(acc_test * 100, 2)) + '\n')
        text_file.write("\n \n" )
        text_file.write("Test set results: \n \n" )
        text_file.write('Macro F1-score: {}'.format(np.round(macro_fscore, 2)) + '\n')
        text_file.write('Average Precision: {}'.format(np.round(avg_precision, 2)) + '\n')
        text_file.write('Average Recall: {}'.format(np.round(avg_recall, 2)) + '\n')
        text_file.write("\n \n" )
        text_file.write("F1-scores per class (test set): \n \n" )
        for class_nr in np.arange(0, len(class_names)):
            text_file.write('{}'.format(class_names[class_nr]) + ': {}'.format(np.round(fscore_classes[class_nr], 2)) + '\n')
        text_file.close()
        
        # save confusion matrix to excel file
        confusion_test = pd.DataFrame(mean_conf_test)
        confusion_test.columns = class_names.values()
        confusion_test.index = class_names
        confusion_test.to_excel(path_out + os.sep + "DNN_confusion_test_" + name_of_step + ".xlsx")
        confusion_train = pd.DataFrame(mean_conf_train)
        confusion_train.columns = class_names.values()
        confusion_train.index = class_names
        confusion_train.to_excel(path_out + os.sep + "DNN_confusion_train_" + name_of_step + ".xlsx")

        np.save(path_out + os.sep + train_name_out + name_of_step + ".npy", mean_conf_train)
        np.save(path_out + os.sep + test_name_out + name_of_step + ".npy", mean_conf_test)


def calc_performance_metrices(subset_name, model, history, trainingset, path_txt, gwl_selection, class_names, perform_metrics, return_f_score = False):
        
    if len(perform_metrics) == 1:
        if subset_name == "test":
            subset_loss, subset_perform1 = model.evaluate(trainingset.x_test, trainingset.y_test_encoded, verbose=0)
            y_pred_class_subset = classify_predictions(trainingset.x_test, model)  # delete binary
        elif subset_name == "dev":
            subset_loss, subset_perform1 = model.evaluate(trainingset.x_dev, trainingset.y_dev_encoded, verbose=0)
            y_pred_class_subset = classify_predictions(trainingset.x_dev, model)  # delete binary
        elif subset_name == "train":
            subset_loss, subset_perform1 = model.evaluate(trainingset.x_train, trainingset.y_train_encoded, verbose=0)
            y_pred_class_subset = classify_predictions(trainingset.x_train, model)  # delete binary
        else:
            raise ValueError("define subset")
        print(subset_name + ' loss:', np.round(subset_loss, 2))
        print(subset_name + " " + perform_metrics[0] + ": [%]", np.round(subset_perform1 * 100, 2))
    elif len(perform_metrics) == 2:
        if subset_name == "test":
            subset_loss, subset_perform1, subset_perform2 = model.evaluate(trainingset.x_test, trainingset.y_test_encoded, verbose=0)
            y_pred_class_subset = classify_predictions(trainingset.x_test, model)  # delete binary
        elif subset_name == "dev":
            subset_loss, subset_perform1, subset_perform2 = model.evaluate(trainingset.x_dev, trainingset.y_dev_encoded, verbose=0)
            y_pred_class_subset = classify_predictions(trainingset.x_dev, model)  # delete binary
        elif subset_name == "train":
            subset_loss, subset_perform1, subset_perform2 = model.evaluate(trainingset.x_train, trainingset.y_train_encoded, verbose=0)
            y_pred_class_subset = classify_predictions(trainingset.x_train, model)  # delete binary
        else:
            raise ValueError("define subset")
        print(subset_name + 'loss:', np.round(subset_loss, 2))
        print(subset_name + " " + perform_metrics[0] + ": [%]", np.round(subset_perform1 * 100, 2))
        print(subset_name + " " + perform_metrics[1] + ": [%]", np.round(subset_perform2 * 100, 2))
    else:
        print("evaluation not defined for more than two performance metrics")
    
    # compute confusion matrix before applying 3-days-rule
    if gwl_selection == "binary":
        confusion_subset = calculate_confusion_matrix_binary("direct-network-output", subset_name, y_pred_class_subset, trainingset, path_txt, perform_metrics, subset_perform1, return_f_score)
    else:
        confusion_subset = calculate_confusion_matrix_multiclass("direct-network-output", subset_name, y_pred_class_subset, trainingset, model, class_names, path_txt)
    
    # save confusion
    np.save(path_txt + os.sep + subset_name + "_confusion_direct-network-output.npy", confusion_subset)     
        
    # computation of performance metrics before 3-day-rule
    if len(perform_metrics) == 1:
        macro_fscore = conf_matrix_to_perform_metrics(confusion_subset, class_names, path_txt, "DNN_metrics_direct-network-output_", subset_name, perform_metrics, subset_loss, subset_perform1)
    elif len(perform_metrics) == 2:
        macro_fscore = conf_matrix_to_perform_metrics(confusion_subset, class_names, path_txt, "DNN_metrics_direct-network-output_", subset_name, perform_metrics, subset_loss, subset_perform1, subset_perform2)

    # Apply 3-days-rule
    if subset_name == "test":
        y_pred_class_subset = apply_three_day_rule(trainingset.x_test, model, y_pred_class_subset)
    if subset_name == "dev":
        y_pred_class_subset = apply_three_day_rule(trainingset.x_dev, model, y_pred_class_subset)
    if subset_name == "train":
        y_pred_class_subset = apply_three_day_rule(trainingset.x_train, model, y_pred_class_subset)  
        
    # compute confusion matrix after applying 3-days-rule
    if gwl_selection == "binary":
        confusion_subset = calculate_confusion_matrix_binary("three-days-rule", subset_name, y_pred_class_subset, trainingset, path_txt, perform_metrics, subset_perform1, return_f_score)
    else:
        confusion_subset = calculate_confusion_matrix_multiclass("three-days-rule", subset_name, y_pred_class_subset, trainingset, model, class_names, path_txt)
        
    np.save(path_txt + os.sep + subset_name + "_confusion_three-days-rule.npy", confusion_subset)      

    # computation of performance metrics
    if len(perform_metrics) == 1:
        macro_fscore = conf_matrix_to_perform_metrics(confusion_subset, class_names, path_txt, "DNN_metrics_three-days-rule_", subset_name, perform_metrics, subset_loss, subset_perform1)
    elif len(perform_metrics) == 2:
        macro_fscore = conf_matrix_to_perform_metrics(confusion_subset, class_names, path_txt, "DNN_metrics_three-days-rule_", subset_name, perform_metrics, subset_loss, subset_perform1, subset_perform2)

    if return_f_score == True and subset_name == "dev":
        return macro_fscore        


def evaluate_dnn(model, history, trainingset, path_txt, path_confusion, gwl_selection, class_names, perform_metrics, eval_test = True, eval_dev = True, return_f_score = False):

    if eval_test == True and eval_dev == True:
        
        subset_names = ["test", "dev", "train"]
        plot_learning_curves(history, path_txt, perform_metrics)
        for s, subset_name in enumerate(subset_names):
            calc_performance_metrices(subset_name, model, history, trainingset, path_txt, gwl_selection, class_names, perform_metrics, return_f_score = False)
                
    elif eval_test == True and eval_dev == False:
        
        subset_names = ["test", "train"]
        for s, subset_name in enumerate(subset_names):
            calc_performance_metrices(subset_name, model, history, trainingset, path_txt, gwl_selection, class_names, perform_metrics, return_f_score = False)

    elif eval_test == False and eval_dev == True:
        
        subset_names = ["dev", "train"]
        plot_learning_curves(history, path_txt, perform_metrics)
        for s, subset_name in enumerate(subset_names):
        
            if return_f_score == True and subset_name == "dev":
                f_score = calc_performance_metrices(subset_name, model, history, trainingset, path_txt, gwl_selection, class_names, perform_metrics, return_f_score = True)
                
            else:
                calc_performance_metrices(subset_name, model, history, trainingset, path_txt, gwl_selection, class_names, perform_metrics, return_f_score = False)
    
    if return_f_score == True:
        return f_score
