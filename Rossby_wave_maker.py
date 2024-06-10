import numpy as np
from netCDF4 import Dataset

# Define parameters
k = 32  # wavenumber in x direction
l = 0  # wavenumber in y direction
beta = 4     # Rossby wave parameter, example value for Earth

# Grid and time parameters
nx = 256   # number of points in x
ny = 256  # number of points in y
nt = 150  # number of time steps
Lx=6.283185307179586




x = np.linspace(0, Lx, nx)
y = np.linspace(0, Lx, ny)
t = np.linspace(0, 100, nt)
T,Y,X = np.meshgrid(t,y,x,indexing='ij')

# Rossby wave dispersion relation to calculate the angular frequency
omega = beta * k / (k**2 + l**2)
print("omega=",omega)
# Calculate the streamfunction
psi = np.sin(k*X + l*Y - omega*T)




# Create a NetCDF file
ncfile = Dataset('rossby_wave.nc', 'w', format='NETCDF4')

# Create dimensions
ncfile.createDimension('x', nx)
ncfile.createDimension('y', ny)
ncfile.createDimension('t', nt)

# Create variables
x_nc = ncfile.createVariable('x', np.float32, ('x',))
y_nc = ncfile.createVariable('y', np.float32, ('y',))
t_nc = ncfile.createVariable('t', np.float32, ('t',))
psi_nc = ncfile.createVariable('psi', np.float32, ('t','y','x'))

# Assign data to variables
x_nc[:] = x
y_nc[:] = y
t_nc[:] = t
psi_nc[:, :, :] = psi

# Add units attributes (optional but recommended)
x_nc.units = 'meters'
y_nc.units = 'meters'
t_nc.units = 'seconds'
psi_nc.units = 'streamfunction'

# Close the NetCDF file
ncfile.close()
