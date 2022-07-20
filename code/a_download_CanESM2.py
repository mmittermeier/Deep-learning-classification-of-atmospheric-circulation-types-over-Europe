"""
Script to Download files from CCCMA Server
http://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CanSISE/output/CCCma/CanESM2/

copyright: Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich
citation: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig,
    R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types
    over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental
    Research Letters. accepted.
@author: m.mittermeier@lmu.de
"""

import os
import numpy as np
from urllib import request
import datetime as dt


def download_psl(remote_url, out_dir, test_download):
    
    # Reproduce directory structure
    run = ["historical-r1", "historical-r2", "historical-r3", "historical-r4", "historical-r5"]
    family = ["r1i1p1", "r2i1p1", "r3i1p1", "r4i1p1", "r5i1p1", "r6i1p1", "r7i1p1", "r8i1p1", "r9i1p1", "r10i1p1"]
    temporal_resolution = "day"
    esm_part = "atmos"
    variable = "psl"
    
    # Create datestring for filename
    year_start = ['1950', '2021']
    monthday_start = "0101"
    year_end = ['2020', '2100']
    monthday_end = "1231"
    
    # Create list with file specific urls
    url_filepaths = []
    for i in run:
        # Loop over runs
        url_run = remote_url + i + "/"
    
        # Loop over families for desired variable
        for j in family:
            url_var_dir = url_run + temporal_resolution + "/" + esm_part + "/" + variable + "/" + j + "/"
    
            # Create filenames for each subdirectory
            # Filename structure: zg_day_CanESM2_historical-r1_r1i1p1_19500101-19551231.nc
            for k, l in zip(year_start, year_end):
                url_filepath = url_var_dir + variable + "_" + temporal_resolution + "_" + "CanESM2" + "_" \
                               + i + "_" + j + "_" + str(k) + monthday_start + "-" + str(l) + monthday_end + ".nc"
                url_filepaths.append(url_filepath)
    
    print("done: get all psl download urls")

    # Download files
    for m in url_filepaths:
        # Split url to create directory on disk
        path_split = m.split("/")
        outpathdir = os.path.join(out_dir, path_split[9], variable, path_split[13])
        if not os.path.exists(outpathdir):
            os.makedirs(outpathdir)
    
        # Get full local path of download file
        outfilepath = os.path.join(outpathdir, path_split[-1])
        print("downloading file: " + path_split[-1])
        print("Download start at: " + dt.datetime.now().strftime("%H:%M:%S"))
    
        # Download file
        request.urlretrieve(m, outfilepath)

        # If test_download is True, only download one file for to save time
        if test_download is True:
            break
    
    print("download of psl data finished")
    
#####################################################################################################
#####################################################################################################


def download_zg(remote_url, out_dir, test_download):
        
    # Reproduce directory structure
    run = ["historical-r1", "historical-r2", "historical-r3", "historical-r4", "historical-r5"]
    family = ["r1i1p1", "r2i1p1", "r3i1p1", "r4i1p1", "r5i1p1", "r6i1p1", "r7i1p1", "r8i1p1", "r9i1p1", "r10i1p1"]
    temporal_resolution = "day"
    esm_part = "atmos"
    variable = "zg"

    # Create datestring for filename
    year_start = np.arange(1951, 2100, 5).tolist()
    year_start = [1950 if x == 1951 else x for x in year_start]
    monthday_start = "0101"
    year_end = np.arange(1955, 2101, 5).tolist()
    monthday_end = "1231"

    # Create list with file specific urls
    url_filepaths = []
    for i in run:
        # Loop over runs
        url_run = remote_url + i + "/"

        # Loop over families for desired variable
        for j in family:
            url_var_dir = url_run + temporal_resolution + "/" + esm_part + "/" + variable + "/" + j + "/"

            # Create filenames for each subdirectory
            # Filename structure: zg_day_CanESM2_historical-r1_r1i1p1_19500101-19551231.nc
            for k, l in zip(year_start, year_end):
                url_filepath = url_var_dir + variable + "_" + temporal_resolution + "_" + "CanESM2" + "_" \
                               + i + "_" + j + "_" + str(k) + monthday_start + "-" + str(l) + monthday_end + ".nc"
                url_filepaths.append(url_filepath)

    print("done: get all zg download urls")

    # Download files
    for m in url_filepaths:
        # Split url to create directory
        path_split = m.split("/")
        outpathdir = os.path.join(out_dir, path_split[9], variable, path_split[13])
        if not os.path.exists(outpathdir):
            os.makedirs(outpathdir)

        # Get full local path of download file
        outfilepath = os.path.join(outpathdir, path_split[-1])
        print("downloading file: " + path_split[-1])
        print("Download start at: " + dt.datetime.now().strftime("%H:%M:%S"))

        # Download file
        request.urlretrieve(m, outfilepath)

        # If test_download is True, only download one file to save time
        if test_download is True:
            break

    print("download of zg data finished")
