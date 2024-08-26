import netCDF4 as nc
import numpy as np

# Open the original netCDF file
source_file = '/home/massoale/Simu_Test/simu_dahu/simu_dahu901/outdir_0001/vars.nc'
src = nc.Dataset(source_file, 'r')


# Check the dimensions and variables
print(src)

# Extract the last time index
last_time_index = len(src.dimensions['time']) - 1

# Create a new netCDF file
dest_file = 'restart.nc'
dst = nc.Dataset(dest_file, 'w', format='NETCDF4')

# Define the dimensions
dst.createDimension('time', None)
dst.createDimension('level', len(src.dimensions['level']))
dst.createDimension('y', len(src.dimensions['y']))
dst.createDimension('x', len(src.dimensions['x']))

# Define the variables and copy attributes
for name, variable in src.variables.items():
    var = dst.createVariable(name, variable.datatype, variable.dimensions)
    var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})

# Copy the data for the last time step
dst.variables['time'][0] = src.variables['time'][last_time_index]
dst.variables['level'][:] = src.variables['level'][:]
dst.variables['y'][:] = src.variables['y'][:]
dst.variables['x'][:] = src.variables['x'][:]
dst.variables['psi'][0, :, :, :] = src.variables['psi'][last_time_index, :, :, :]
dst.variables['q'][0, :, :, :] = src.variables['q'][last_time_index, :, :, :]

# Close the files
src.close()
dst.close()

print(f"Created new file {dest_file} with the last snapshot.")
