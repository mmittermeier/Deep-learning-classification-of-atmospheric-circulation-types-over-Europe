"""
Script to subset PSL from global CanESM2 Data to GWL Domain

copyright: Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich
citation: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig,
    R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types
    over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental
    Research Letters. accepted.
@author: m.mittermeier@lmu.de
"""

import xarray as xr
import os
import numpy as np
from os import listdir
from datetime import datetime


def extract_psl(src_file, path, outpath, variable):

    print(src_file)

    # Get reference file
    src_data = xr.open_dataset(src_file)
    src_lat = src_data['lat']
    src_lon = src_data['lon']
    
    # Loop over all files and append the same ones to a 80 year file
    run = os.listdir(path)
    family = os.listdir(os.path.join(path, run[0], variable))
    
    # Loop over subdirectories seperately
    for r in run:
        # Change cd to run subdirectory
        run_path = os.path.join(path, r, variable)
    
        for f in family:
    
            # Get file list
            # Change cd to family subdirectory based on run directory
            family_path = os.path.join(run_path, f)
            # Get all files in directory
            files = [k for k in listdir(family_path)]
    
            # Create outfilename
            # Take first file to create outfile name from it
            namefile = files[0]
            # Cut off date in filename (which makes filename unique)
            outfilename = namefile[0:-21]
            # Create outfile directory representing the infile directory strucutre: Out_Folder/Run/Family/Out_File
            outfiledir = os.path.join(outpath, r, f)
            # If directory does not exist, create it
            if not os.path.exists(outfiledir):
                os.makedirs(outfiledir)
    
            # Create two data bins one for 1950-2020 and one for 2021-2100 period
            ds_1 = []
            ds_2 = []
    
            # Loop over all files in subdirectory
            for d in files:
    
                # Print status update
                print(d)
    
                # Get start year of file
                year_start = d.split("_")[-1][0:4]
    
                # If year start is below 2020 append it to data bin 1 (ds_1)
                if int(year_start) <= 2020:
    
                    # Open data
                    psl_data = xr.open_dataset(os.path.join(family_path, d))
    
                    # Get time variable
                    psl_time = psl_data['time']
    
                    # Select NAA + EU domain by coordinates
                    # Boundaries calculated from corner coordinates of src_data.lon and src_data.lat
                    # Create two variables for east (lon>0) and west(lon<0)
                    # Values for src_data.lon: min: -63.562874, max: 43.312801 --> rounded to 295 for min (resulting from 360 - 63.562874) and 43.4 for max
                    # Values for src_data.lat: min: 31.731627 , max: 73.853111 --> rounded to 31.6 for min and to 74 for max
                    psl_e = psl_data['psl'].sel(lat=slice(31.6, 74)).sel(lon=slice(0, 43.4))
                    psl_w = psl_data['psl'].sel(lat=slice(31.6, 74)).sel(lon=slice(295.0, 360.0))

                    # Convert to numpy arrays
                    psl_e_arr = np.asarray(psl_e)
                    psl_w_arr = np.asarray(psl_w)
    
                    # Concat east and west in new variable
                    psl = np.concatenate((psl_w_arr, psl_e_arr), axis=2)

                    # Create xarray dataset with coordinates from src_data
                    ds = xr.Dataset(
                        data_vars=dict(
                            psl=(["time", "x", "y"], psl)
                        ),
                        coords=dict(
                            lon=(["x", "y"], src_lon),
                            lat=(["x", "y"], src_lat),
                            time=psl_time
                        ),
                        attrs=dict(description="CanESM_downsized")
                    )
                    ds_1.append(ds)
    
                # If year start is above 2020 append it to data bin 2 (ds_2)
                else:
    
                    # Open data
                    psl_data = xr.open_dataset(os.path.join(family_path, d))
    
                    # Get time variable
                    psl_time = psl_data['time']
    
                    # Select NAA + EU domain by coordinates
                    # Boundaries calculated from corner coordinates of src_data.lon and src_data.lat
                    # Create two variables for east (lon>0) and west(lon<0)
                    # Values for src_data.lon: min: -63.562874, max: 43.312801 --> rounded to 295 for min (resulting from 360 - 63.562874) and 43.4 for max
                    # Values for src_data.lat: min: 31.731627 , max: 73.853111 --> rounded to 31.6 for min and to 74 for max
                    psl_e = psl_data['psl'].sel(lat=slice(31.6, 74)).sel(lon=slice(0, 43.4))
                    psl_w = psl_data['psl'].sel(lat=slice(31.6, 74)).sel(lon=slice(295.0, 360.0))
    
                    # Convert to numpy arrays
                    psl_e_arr = np.asarray(psl_e)
                    psl_w_arr = np.asarray(psl_w)
    
                    # Concat east and west in new variable
                    psl = np.concatenate((psl_w_arr, psl_e_arr), axis=2)
    
                    # Create xarray dataset with coordinates from src_data
                    ds = xr.Dataset(
                        data_vars=dict(
                            psl=(["time", "x", "y"], psl)
                        ),
                        coords=dict(
                            lon=(["x", "y"], src_lon),
                            lat=(["x", "y"], src_lat),
                            time=psl_time
                        ),
                        attrs=dict(description="CanESM_downsized")
                    )
                    ds_2.append(ds)
    
            # Concat files to one dataframe for two desired time periods
            ds_1_combined = xr.concat(ds_1, dim='time')
            ds_1_combined.to_netcdf(os.path.join(outfiledir, outfilename + "_19500101-20201231.nc"))
            if ds_2:  # True if ds_2 list is not empty; i.e. if only one test file is downloaded
                ds_2_combined = xr.concat(ds_2, dim='time')
                ds_2_combined.to_netcdf(os.path.join(outfiledir, outfilename + "_20210101-21001231.nc"))
    
    print("done extracting psl data")
    
    
def extract_z500(src_file, path, outpath, variable):
    
    # Get reference file
    ref_file_data = xr.open_dataset(src_file)
    ref_file_lat = ref_file_data['lat']
    ref_file_lon = ref_file_data['lon']
    
    # Loop over all files and append the same ones to a 80 year file
    run = os.listdir(path)
    family = os.listdir(os.path.join(path, run[0], variable))

    # Loop over subdirectories seperately
    for r in run:
        # Change cd to run subdirectory
        run_path = os.path.join(path, r, variable)
    
        for f in family:
            # Get file list
            # Change cd to family subdirectory based on run directory
            family_path = os.path.join(run_path, f)
            # Get all files in directory
            files = [k for k in listdir(family_path)]
    
            # Create outfilename
            # Take first file to create outfile name from it
            namefile = files[0]
            # Cut off date in filename (which makes filename unique)
            outfilename = namefile[0:-21]
            # Create outfile directory representing the infile directory strucutre: Out_Folder/Run/Family/Out_File
            outfiledir = os.path.join(outpath, r, f)

            # If directory does not exist, create it
            if not os.path.exists(outfiledir):
                os.makedirs(outfiledir)
    
            # Create two data bins one for 1950-2020 and one for 2021-2100 period
            ds_1 = []
            ds_2 = []

            # Loop over files in subdirectory
            for d in sorted(files):
    
                # Print status update
                print(d)
    
                # Get start year of file
                year_start = d.split("_")[-1][0:4]
    
                # If year start is below 2020 append it to data bin 1 (ds_1)
                if int(year_start) <= 2020:
    
                    # Open data
                    z_data = xr.open_dataset(os.path.join(family_path, d))
    
                    # Get time variable
                    z_time = z_data['time']
    
                    # Select NAA + EU domain by coordinates
                    # Boundaries calculated from corner coordinates of src_data.lon and src_data.lat
                    # Create two variables for east (lon>0) and west(lon<0)
                    # Values for src_data.lon: min: -63.562874, max: 43.312801 --> rounded to 295 for min (resulting from 360 - 63.562874) and 43.4 for max
                    # Values for src_data.lat: min: 31.731627 , max: 73.853111 --> rounded to 31.6 for min and to 74 for max
                    z500_e = z_data['zg'].sel(plev=50000).sel(lat=slice(31.6, 74)).sel(lon=slice(0, 43.4))
                    z500_w = z_data['zg'].sel(plev=50000).sel(lat=slice(31.6, 74)).sel(lon=slice(295.0, 360.0))
    
                    # Convert to numpy arrays
                    z500_e_arr = np.asarray(z500_e)
                    z500_w_arr = np.asarray(z500_w)
    
                    # Concat east and west in new variable
                    z500 = np.concatenate((z500_w_arr, z500_e_arr), axis=2)
    
                    # Create xarray dataset with coordinates from src_data
                    ds = xr.Dataset(
                        data_vars=dict(
                            z500=(["time", "x", "y"], z500)
                        ),
                        coords=dict(
                            lon=(["x", "y"], ref_file_lon),
                            lat=(["x", "y"], ref_file_lat),
                            time=z_time
                        ),
                        attrs=dict(description="CanESM_downsized")
                    )
                    ds_1.append(ds)
    
                # If year start is above 2020 append it to data bin 2 (ds_2)
                else:
    
                    # Open data
                    z_data = xr.open_dataset(os.path.join(family_path, d))
    
                    # Get time variable
                    z_time = z_data['time']
    
                    # Select NAA + EU domain by coordinates
                    # Boundaries calculated from corner coordinates of src_data.lon and src_data.lat
                    # Create two variables for east (lon>0) and west(lon<0)
                    # Values for src_data.lon: min: -63.562874, max: 43.312801 --> rounded to 295 for min (resulting from 360 - 63.562874) and 43.4 for max
                    # Values for src_data.lat: min: 31.731627 , max: 73.853111 --> rounded to 31.6 for min and to 74 for max
                    z500_e = z_data['zg'].sel(plev=50000).sel(lat=slice(31.6, 74)).sel(lon=slice(0, 43.4))
                    z500_w = z_data['zg'].sel(plev=50000).sel(lat=slice(31.6, 74)).sel(lon=slice(295.0, 360.0))
    
                    # Convert to numpy arrays
                    z500_e_arr = np.asarray(z500_e)
                    z500_w_arr = np.asarray(z500_w)
    
                    # Concat east and west in new variable
                    z500 = np.concatenate((z500_w_arr, z500_e_arr), axis=2)
    
                    # Create xarray dataset with coordinates from src_data
                    ds = xr.Dataset(
                        data_vars=dict(
                            z500=(["time", "x", "y"], z500)
                        ),
                        coords=dict(
                            lon=(["x", "y"], ref_file_lon),
                            lat=(["x", "y"], ref_file_lat),
                            time=z_time
                        ),
                        attrs=dict(description="CanESM_downsized")
                    )
                    ds_2.append(ds)
    
            # Concat files to one dataframe for two desired time periods
            ds_1_combined = xr.concat(ds_1, dim='time')

            # Select start and end date for naming output file with correct years
            start_date = np.array(ds_1_combined['time'][0])
            str_start_date = str(start_date)[:10].replace('-', '')
            end_date = np.array(ds_1_combined['time'][-1])
            str_end_date = str(end_date)[:10].replace('-', '')

            ds_1_combined.to_netcdf(outfiledir+os.sep+outfilename+"_"+str_start_date+"-"+str_end_date+".nc")
            if ds_2:  # True if ds_2 list is not empy; i.e. if only one test file is downloaded
                ds_2_combined = xr.concat(ds_2, dim='time')

                # Select start and end date for naming output file with correct years
                start_date = np.array(ds_2_combined['time'][0])
                str_start_date = str(start_date)[:10].replace('-', '')
                end_date = np.array(ds_2_combined['time'][-1])
                str_end_date = str(end_date)[:10].replace('-', '')
                ds_2_combined.to_netcdf(outfiledir+os.sep+outfilename+"_"+str_start_date+"-"+str_end_date+".nc")

    print("done extracting zg data")
