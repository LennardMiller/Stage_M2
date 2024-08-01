import xarray as xr
import numpy as np
from scipy.ndimage import zoom

# Load the netCDF file
ds = xr.open_dataset('/home/massoale/Simu_Test/simu_dahu/simu_dahu517/outdir_0001/vars.nc')

# Define the target resolution
target_x, target_y = 160, 160

# Function to interpolate data
def interpolate_data(data, target_shape):
    zoom_factors = [target_shape[0] / data.shape[-2], target_shape[1] / data.shape[-1]]
    return zoom(data, (1, 1, zoom_factors[0], zoom_factors[1]), order=1)  # linear interpolation

# Interpolate the variables
psi_interpolated = interpolate_data(ds['psi'].values, (target_y, target_x))
q_interpolated = interpolate_data(ds['q'].values, (target_y, target_x))

# Create a new dataset with interpolated data
new_ds = xr.Dataset(
    {
        'psi': (('time', 'level', 'y', 'x'), psi_interpolated),
        'q': (('time', 'level', 'y', 'x'), q_interpolated)
    },
    coords={
        'time': ds['time'].values,
        'level': ds['level'].values,
        'x': np.linspace(ds['x'].min(), ds['x'].max(), target_x),
        'y': np.linspace(ds['y'].min(), ds['y'].max(), target_y)
    }
)

# Copy global attributes from the original dataset
new_ds.attrs = ds.attrs

# Save the new dataset
new_ds.to_netcdf('/home/massoale/Bureau/simu_downsized/517_downsized.nc')
