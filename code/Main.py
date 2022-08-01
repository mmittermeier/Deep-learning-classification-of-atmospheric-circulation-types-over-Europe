# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:55:23 2021

Des.: setup script for the application of the deep learning classifier to new climate model data
publication: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig, R. (2022): 
    A Deep Learning based Classification of Atmospheric Circulation Types over Europe: 
    Projection of Future Changes in a CMIP6 Large Ensemble. 
    Environmental Research Letters. 17. doi: 10.1088/1748-9326/ac8068.
    URL: https://iopscience.iop.org/article/10.1088/1748-9326/ac8068
Necessary data:
    following variables of the climate model to which the classifier is to be applied:
    (for demonstration/testing purposes CanESM2 data for these variables can be downloaded with this script)
    - variable sea level pressure (psl)
    - variable geopotential height at 500 hPa (z500)
    in daily resolution over the domain: 30-75°N, -65-45°O, which covers Europe and parts of the North Atlantic 
Example: this script uses one member of the CanESM2-LE (Canadian Earth System Model version 2) as an example
    CanESM2 data is downloaded from following server of the CCCma: http://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CanSISE/output/CCCma/CanESM2/
    for the application to other climate datasets: first download the required data; then disable the first step "run_download_CanESM2"
Python requirements:
    - python version 3.6
    - packages: numpy, xarray (version 0.15.1. or later), os, fnmatch, xesmf, pickle, urllib, datetime, sklearn, matplotlib, pandas, sys
    - note: run_regrid works only on UNIX-systems (Linux, DOS)
Usage: please define your working directory at the beginning containing the reference file for regridding and
       all the code in a subdirectory named "code"
Details:
    1. run_download_CanESM2
        downloads test data of the CanESM2 (global); 3D-variable zg, which contains z500
    2. run_regrid
        2.1. clips global data to required domain (30-75°N, -65-45°O)
        2.2. extracts z500 from 3D-variable zg (works only for data structured like this: time x levels x lat x lon)
        2.3. regrid data to required grid with spatial resolution of 5 x 5 degrees
    3. run_inference
        applies the trained models of the deep ensemble to the new climate model input
        output: circulation type classifications for the daily input data
Acknowledgements: biggest thanks goes to Aaron Banze and Julia Miller from the LMU for their support      
    
copyright: Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich
citation: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig,
    R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types
    over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental
    Research Letters. 17. doi: 10.1088/1748-9326/ac8068.
@author: m.mittermeier@lmu.de
"""

import os
import numpy as np

### START MODIFYING ###

# Working directory should contain the reference file and subfolders "trained_deep_ensemble" and "code"
# Define your working directory here:
wdir = ""
code_dir = os.path.join(wdir, "code")

# For testing purposes, set to True to only download minimal data for faster run time
test_download = True

# General variable definitions
name_climate_model = "CanESM2"
ssp = "historical"
years = "1950-2020"

# Process I. Download data
# Directory to store downloaded, unedited climate data
if not os.path.exists(os.path.join(wdir, "original")):
    os.makedirs(os.path.join(wdir, "original"))
download_dir = os.path.join(wdir, "original")
# Directory for clipped files and extracted z500 variable
if not os.path.exists(os.path.join(wdir, "corrected")):
    os.makedirs(os.path.join(wdir, "corrected"))
corr_dir = os.path.join(wdir, "corrected")

# Process II. Regrid data
ref_file_name = "slp_era40_1957-12-01_to_1957-12-31_dailymean_regridded.nc"  # Reference file used for regridding
z500_level = 3  # Pressure level of z500 in 3D-variable zg of climate model

# Directory to save regridded data
if not os.path.exists(os.path.join(wdir, "regrid")):
    os.makedirs(os.path.join(wdir, "regrid"))
regrid_dir = os.path.join(wdir, "regrid")

# Process III. Inference
deep_ensemble_path = os.path.join(wdir, "trained_deep_ensemble")  # path to the 30 trained networks of the deep ensemble
var_name_slp = "psl"  # name of variable sea level pressure in regridded netcdf file
var_name_z500 = "z500"  # name of variable geopotential height at 500 hPa in regridded netcdf file

# Directory to save output of inference
if not os.path.exists(os.path.join(wdir, "output_inference")):
    os.makedirs(os.path.join(wdir, "output_inference"))
path_inference = os.path.join(wdir, "output_inference")

### END MODIFYING ###


# General definitions (usually shouldn´t be changed by the user)
class_names = ["WA", "WZ", "WS", "WW", "SWA", "SWZ", "NWA", "NWZ", "HM", "BM", "TM", "NA", "NZ", "HNA", "HNZ", "HB",
               "TRM", "NEA", "NEZ", "HFA", "HFZ", "HNFA", "HNFZ", "SEA", "SEZ", "SA", "SZ", "TB", "TRW"]
network_version = "v6-2-init"


def run_inference_process(run_download_CanESM2="off", run_regrid="off", run_inference="off"):

    # I. Download CanESM2 data
    if run_download_CanESM2 == "on":
        import os
        import a_download_CanESM2 as download_data

        remote_url = "http://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CanSISE/output/CCCma/CanESM2/"
        
        # Download global CanESM2 data; download 3D variable zg; the download takes about 5 minutes per file
        download_data.download_psl(remote_url, download_dir, test_download)
        download_data.download_zg(remote_url, download_dir, test_download)

        # Extract z500 from 3D-variable zg and clip global data to required domain
        from b_extract_variables import extract_psl, extract_z500

        # Define path to reference file
        src_file = os.path.join(wdir, ref_file_name)

        extract_psl(src_file, download_dir, corr_dir, "psl")
        extract_z500(src_file, download_dir, corr_dir, "zg")

    # ---------------------------------------------------------------------------------------------------------
    # II. Regrid data
    
    if run_regrid == "on":
        
        from c_regridding import regridding_universal
        import os

        # regrid z500
        regridding_universal(dir_files=corr_dir,
                             pattern="zg_*.nc",
                             path_reference_file=wdir+os.sep+ref_file_name,
                             out_path=regrid_dir,
                             var_name="z500",
                             lon_name="lon",
                             lat_name="lat")

        # regrid psl
        regridding_universal(dir_files=corr_dir,
                             pattern="psl_*.nc",
                             path_reference_file=wdir + os.sep + ref_file_name,
                             out_path=regrid_dir,
                             var_name="psl",
                             lon_name="lon",
                             lat_name="lat")

    # ---------------------------------------------------------------------------------------------------------
    # # III. Inference
    
    if run_inference == "on":

        import os
        from d_inference import inference_universal
        from f_deep_ensemble import run_ensemble_weighted_mean

        # Loop over all subfolders in "/trained_deep_ensemble" containing 30 deep ensemble members
        for n in np.arange(1, 31):
            network_path = os.path.join(deep_ensemble_path, "1_final-model_ppc_"+network_version+str(n))
            inference_universal(
                wdir=code_dir,
                network_path=network_path,
                model_name="trained_model_entire_trainingset.h5",
                dir_files=regrid_dir,
                file_pattern="psl_*.nc",
                pendant_filename_start="zg",
                var_name_slp=var_name_slp,
                var_name_z500=var_name_z500,
                out_path=path_inference+os.sep+"1_final-model_ppc_"+network_version+str(n),
                name_dataset=name_climate_model)

        # Calculate deep-ensemble mean
        run_ensemble_weighted_mean(
            path_results=path_inference,
            path_confusion=deep_ensemble_path,
            path_out=path_inference,
            name_dataset=name_climate_model,
            period=ssp,
            years=years,
            member="1_final-model_ppc_"+network_version,
            num_inits=30,
            network_version=network_version,
            class_names=class_names)


# Execute function: indicate, which process should be executed by setting them to on
### START MODIFY INPUT ###

run_inference_process(run_download_CanESM2="on",
                      run_regrid="on",
                      run_inference="on")

### END MODIFY ###
