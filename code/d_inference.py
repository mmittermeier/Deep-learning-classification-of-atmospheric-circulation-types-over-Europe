"""
Functions to infere regridded files with Network/ ML_Model

Content:
1. Standardize new Data
2. onehot decoding
3. Inference

copyright: Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich
citation: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig,
    R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types
    over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental
    Research Letters. 17. doi: 10.1088/1748-9326/ac8068.
@author: m.mittermeier@lmu.de
"""


def standardize_newdata(X_data, path_network):
    # standardize input data (z-transformation)


    import pickle
    import numpy as np
    import os

    # Load mean and variance of training
    m = np.load(path_network + os.sep + 'standardization' + os.sep + 'mean_X.npy', allow_pickle=True)
    v = np.load(path_network + os.sep + 'standardization' + os.sep + 'var_X.npy', allow_pickle=True)

    # Function for z-transformation
    def normalize_with_moments(X, m, v, epsilon):
        X_normed = np.zeros([X.shape[0], X.shape[1], X.shape[2], X.shape[3]])
        for ex_nr in np.arange(0, X.shape[0]):
            X_normed[ex_nr, :, :] = (X[ex_nr, :, :] - m) / np.sqrt(v + epsilon)  # epsilon to avoid dividing by zero
        return X_normed

    # Apply z-transformation with mean and variance to new data
    X_data_norm = normalize_with_moments(X_data, m, v, epsilon=1e-8)

    # File is not saved but only returned
    return X_data_norm


def onehot_decoding(labels_encoded, m):

    import numpy as np

    labels_decoded = np.zeros([m])
    for x in np.arange(0, m):
        if labels_encoded[x, 0] == 1 and labels_encoded[x, 1] == 0:
            labels_decoded[x] = 0
        elif labels_encoded[x, 0] == 0 and labels_encoded[x, 1] == 1:
            labels_decoded[x] = 1

    return labels_decoded


def inference_universal(wdir, network_path, model_name,
                        dir_files, file_pattern, pendant_filename_start,
                        var_name_slp, var_name_z500, out_path, name_dataset):
    """
    Funktion to calculate inference of network and data

    Parameters:
        - gwl_selection: either "all", "BM" or "multiple"
        - network_path: path to network file, depending on machine and gwl selection
        - best_model_name: name of the model to use
        - dir_files: parent directory of files on which network is applied
        - file_pattern: pattern for slp files in directory
        - pendant_filename_start: start of z500 filename which is read in according to slp file, usually filname starts with variable name, e. g. "zg", "tas"
        - var_name_slp: variable name for sea level pressure, varies between slp, mslp, msl, psl or __xarray_dataarray_variable__
        - var_name_z500: variable name for pressure at 500m
        - out_path: directory of output inference files
    """
    import os
    from fnmatch import fnmatch
    import netCDF4 as nc
    import numpy as np
    from tensorflow import keras
    import sys
    
    sys.path.insert(1, wdir)
    import e_evaluate_dnn as B_eval
    
    # Load network
    best_network = keras.models.load_model(os.path.join(network_path, model_name))
    best_network.summary()

    # Get all files in path
    # s. function FilesInDir in fx_CanESM2_lon_bounds
    filelist = []
    fulllist = []

    for path, subdirs, files in os.walk(dir_files):
        for name in files:
            if fnmatch(name, file_pattern):
                filelist.append(name)
                fulllist.append(os.path.join(path, name))

    # Loop over files
    for full, file in zip(fulllist, filelist):

        # Get pendant with second variable "z500"
        file_split = full.split(os.sep)
        filename_split = file[3:len(file)]

        pendant_file_dir = os.sep.join(file_split[0: len(file_split) - 1])

        # Check if pendant file with zg variable was downloaded and get the complete file name
        # This method accounts for different naming of pendant file if not all years for zg were downloaded
        pendant_file_list = [f for f in os.listdir(pendant_file_dir) if os.path.isfile(os.path.join(pendant_file_dir, f)) and \
                             pendant_filename_start+filename_split[:-19] in f]

        if not pendant_file_list:  # If list is empty, there is no pendant file (not enough data for zg downloaded)
            continue

        # Read the regridded datasets for zg and psl
        ncfile_zg = nc.Dataset(os.path.join(pendant_file_dir, pendant_file_list[0]), "r")
        data_var_zg = np.array(ncfile_zg.variables[var_name_z500][:])
        print("mean z500:", np.mean(data_var_zg))

        ncfile_slp = nc.Dataset(full, "r")

        data_var_slp = np.array(ncfile_slp.variables[var_name_slp][:])

        # Reduce dataset to available years for zg var., e.g. if test_download == True only few years downloaded for zg
        data_var_slp = data_var_slp[:len(data_var_zg), :, :]

        # Create dataset with first variable "slp" and second variable "z500"
        data = np.concatenate(([data_var_slp[:, :, :, np.newaxis], data_var_zg[:, :, :, np.newaxis]]), axis=3)

        # Initialize data
        X_data = standardize_newdata(data, network_path)

        # Inference
        pred_prob = best_network.predict(X_data)  # derive probabilities
        y_pred_class = B_eval.classify_predictions(X_data, best_network)
                
        # Apply 3-days-rule
        y_pred_class_3dr = B_eval.apply_three_day_rule(X_data, best_network, y_pred_class)  

        # Save files
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        np.save(out_path + os.sep + "Inference_" + name_dataset + "_all_y_pred_3dr_" + pendant_file_list[0][3:-3] + ".npy", y_pred_class_3dr, allow_pickle=True)
        np.save(out_path + os.sep + "Probability_" + name_dataset + "_all_y_prob_before-3dr_" + pendant_file_list[0][3:-3] + ".npy", pred_prob)

    print("done", out_path)
