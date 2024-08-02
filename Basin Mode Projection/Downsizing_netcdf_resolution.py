import xarray as xr

# Open the original dataset with chunking (optional but helpful for large files)
ds = xr.open_dataset('/home/massoale/Simu_Test/simu_dahu/simu_dahu920/outdir_0001/vars.nc', chunks={'time': 1, 'y': 2049, 'x': 2049})

# Downscale to 256x256 using coarsen
ds_coarsened = ds.coarsen(x=8, y=8, boundary='trim').mean()

# Verify new dimensions
print(ds_coarsened)

# Save the downscaled dataset
ds_coarsened.to_netcdf('/home/massoale/Simu_Test/simu_dahu/simu_dahu920/outdir_0001/vars_256x256.nc')


