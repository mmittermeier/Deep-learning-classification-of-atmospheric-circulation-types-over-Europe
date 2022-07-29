# Deep-learning-classification-of-atmospheric-circulation-types-over-Europe
This repository accompanies the publication "A Deep Learning based Classification of Atmospheric Circulation Types over Europe: Projection of Future Changes in a CMIP6 Large Ensemble" in Environmental Research Letters. This repository provides the code to apply the trained deep ensemble described in this publication to climate model data: https://iopscience.iop.org/article/10.1088/1748-9326/ac8068

## Citation:
Please use following citation to cite this work: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig, R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental Research Letters. 17. doi: 10.1088/1748-9326/ac8068.

## Python requirements:
    - python version 3.6
    - packages: numpy, xarray (version 0.15.1. or later), os, fnmatch, xesmf
    - note: "regridding_universial" in "c_regridding" only runs on UNIX-systems (Linux, DOS)

## Necessary data:
* For demonstration/testing purposes CanESM2-LE data can be downloaded using this code
* Otherwise following variables are needed for a specific climate model, to which the classifier is to be applied:
    - sea level pressure (psl)
    - geopotential height at 500 hPa (z500)
* temporal resolution: daily 
* domain: 30-75°N, -65-45°O (covers Europe and parts of the North Atlantic)
* example: this script uses one member of the CanESM2-LE (Canadian Earth System Model version 2) as an example
    CanESM2 data is downloaded from following server of the CCCma: http://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CanSISE/output/CCCma/CanESM2/
    for the application to other climate datasets: first download the required data; then disable the first step "run_download_CanESM2"
    
    
## Usage: 
please define your working directory in "Main.py". The working directory should contain the reference file for regridding and
all the code in a subdirectory named "code"

## Details:
    1. run_download_CanESM2
        downloads test data of the CanESM2-LE (global); "slp" and 3D-variable "zg", which contains z500
    2. run_regrid
        2.1. clips global data to required domain (30-75°N, -65-45°O)
        2.2. extracts z500 from 3D-variable zg (works only for data structured like this: time x levels x lat x lon)
        2.3. regrids data to the required grid, on which the deep ensemble was trained on, with spatial resolution of 5 x 5 degrees
    3. run_inference
        applies the trained models of the deep ensemble to the new climate model input
        output: circulation type classifications for the daily input data
        
## Acknowledgements: 
* ERA-40 reanalysis data is derived from the European Centre for Medium-Range Weather Forecasts (ECMWF): www.ecmwf.int/
* biggest thanks goes to Aaron Banze and Julia Miller (LMU) for their support      
    
## Copyright: 
Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich

## contact:
@author: m.mittermeier@lmu.de

