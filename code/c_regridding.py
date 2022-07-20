"""
Functions to regrid files for Inference
environment with xesmf needed (only available on mac or linux)

copyright: Magdalena Mittermeier, Ludwig-Maximilians Universität (LMU) München, Munich
citation: Mittermeier, M., Weigert, M., Rügamer, D., Küchenhoff, H. & Ludwig,
    R. (2022): A Deep Learning based Classification of Atmospheric Circulation Types
    over Europe: Projection of Future Changes in a CMIP6 Large Ensemble. Environmental
    Research Letters. accepted.
@author: m.mittermeier@lmu.de
"""


def regridding_universal(dir_files, pattern, path_reference_file, out_path, var_name, lon_name, lat_name):
    """
    Function to apply regridding on all platforms, with all data sources by all users

    Parameters:
        dir_files: Path to parent directory of files to be regridded. Loop goes recursively over list of files in all subdirectories
        pattern: file name pattern, depending on downloaded variable name zg/psl
        path_reference_file: file containing the desired grid information and to which dir_files are regridded to.
        out_path: Parent directory to where regridded files are saved
        var_name: variable name on which regridding is applied
        lon_name: name of longitudes variable/ coordinate in file to be regridded
        lat_name: name of latitude variable/ coordinate in file to be regridded
    """

    import xarray as xr
    import os
    from fnmatch import fnmatch
    import xesmf as xe

    # Get dir file length
    dir_file_length = len(dir_files.split('/'))

    # Get all files to be regridded
    # s. function FilesInDir in fx_CanESM2_lon_bounds
    filelist = []
    fulllist = []
    for path, subdirs, files in os.walk(dir_files):
        for name in files:
            if fnmatch(name, pattern):
                filelist.append(name)
                fulllist.append(os.path.join(path, name))

    # Loop over files
    for f in fulllist:
    
        # Load reference file - aggregated climex file as xarray
        reference_data = xr.open_dataset(path_reference_file)
        #reference_data = reference_data.rename({'latitude': 'lat', 'longitude': 'lon'})
        
        # Load era-20c file as xarray
        regrid_data = xr.open_dataset(f)

        # Rename coordinates if applicable
        if lon_name == 'longitude':
            regrid_data = regrid_data.rename({'longitude': 'lon'})

        if lat_name == 'latitude':
            regrid_data = regrid_data.rename({'latitude': 'lat'})

        # Get variable
        regrid_var = regrid_data[var_name]

        # Regrid era to climex-agg-flip
        regridder = xe.Regridder(regrid_data, reference_data, 'bilinear', reuse_weights=False)
        regrid_out = regridder(regrid_var)

        # Create out directory and outfile from parent indirectory
        # Split filename
        path_split = f.split('/')

        # Exclude Parent Infile Directory and File name in path_subdirs
        path_subdirs = path_split[dir_file_length:len(path_split)-1]

        # Merge elements of split to one string
        out_subpath = '/'.join(path_subdirs)

        # Generate out file name by cutting off file ending and appending new name and ending
        out_filename = path_split[-1][:-3]
        out_filename = out_filename + "_regrid.nc"
        if not os.path.exists(out_path + os.sep + out_subpath):
            os.makedirs(out_path + os.sep + out_subpath)
        print("save regrid output to: \n", out_path + os.sep + out_subpath + os.sep + out_filename)

        # Save to netcdf
        regrid_out.to_netcdf(
            out_path + os.sep + out_subpath + os.sep + out_filename,
            "w")
