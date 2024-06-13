import xarray as xr

# Load the NetCDF file
file_path = "/home/massoale/Simu_Test/simu_dahu/simu_dahu419/outdir_0001/vars.nc"  # Replace this with the path to your file
data = xr.open_dataset(file_path)

# Slice the dataset to keep only the first 100 time steps
data_resized = data.isel(time=slice(0, 160))

# Save the resized dataset to a new file
output_file_path = "/home/massoale/Simu_Test/simu_dahu/simu_dahu219/outdir_0001/vars.nc"
data_resized.to_netcdf(output_file_path)
