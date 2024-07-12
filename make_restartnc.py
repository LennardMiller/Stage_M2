import netCDF4 as nc

# # Open the input NetCDF file
# simu_number=input("What is the number of simulation you want to extract the last snapshot? ")

# if simu_number.isdigit():
#     input_file = '/home/massoale/Simu_Test/simu_dahu/simu_dahu'+str(simu_number)+'/outdir_0001/vars.nc'
# else:
#     print("Please enter a number")
input_file ='/home/massoale/Bureau/Stage_M2/vars_Mardi.nc'
output_file = 'restart.nc'

# Open the input NetCDF file
with nc.Dataset(input_file, 'r') as src:
    # Print all variables in the file
    print("Variables in the file:")
    for var_name in src.variables.keys():
        print(var_name)
    
    # Create a new NetCDF file to save the last snapshot
    with nc.Dataset(output_file, 'w') as dst:
        # Copy dimensions from the original file to the new file
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
            
        # Copy variables from the original file to the new file
        for name, variable in src.variables.items():
            # Create a new variable in the destination file
            new_var = dst.createVariable(name, variable.datatype, variable.dimensions)
            new_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            
            # Copy data for the last time step for variables with the time dimension
            if 'time' in variable.dimensions:
                if len(variable.shape) == 4:  # Assuming the variable has shape (time, level, y, x)
                    if name == 'psi':
                        new_var[0, 0, :, :] = variable[2, 0, :, :]/1e9
                        psi=new_var[0, 0, :, :]
                        print("psi is updated")
                    elif name =='q':
                        new_var[0, 0, :, :] = psi*8.458997098232025e-06**2
                        print("q is updated")
                    else:
                        new_var[0, :, :, :] = variable[2, :, :, :]
                elif len(variable.shape) == 3:  # For variables with shape (time, y, x) if any
                    new_var[0, :, :] = variable[2, :, :]
                elif len(variable.shape) == 2:  # For variables with shape (time, level) if any
                    new_var[0, :] = variable[2, :]
                else:  # For 1D variables with shape (time,)
                    new_var[0] = variable[2]
            else:
                new_var[:] = variable[:]

print(f"Last snapshot saved to {output_file}")


