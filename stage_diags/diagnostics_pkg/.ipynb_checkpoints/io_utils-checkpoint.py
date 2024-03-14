''' collection of function to facilitate input/output '''

import numpy as np
import xarray as xr

def read_params(dir0):
    'reads param.in file into a dictionary'
    
    # create params dictionary 
    params = {}
    
    # read model parameters
    text = open(''.join([dir0, "/params.in" ])).read()
    lines = text.split('\n')
    for line in lines:
        if len(line) != 0:
            if line[0] != '#':
                strings = line.split('=')
                varname = strings[0]
                varvalue = strings[1]
                if varvalue[1] == '[':
                    varvalue = varvalue[2:-1].split(',')
                    varvalue = np.array([float(varvalue[n]) for n in range(len(varvalue))])
                else:
                    varvalue = float(varvalue[1:])
                params.update({varname.strip() : varvalue})
        
    return params

def read_nc(dir0, N_chunk = 5e7, chunkdim = 'time', tconv = 0):
    'reads nc file into a dask array. The N_chunk value works well for my personal computer.'
        
    ds = xr.open_dataset(dir0 + '/vars.nc')
    
    tfin = ds.time[-1].values
    dt_out = tfin - ds.time[-2].values
    
    ds = ds.sel(time=np.linspace(tconv, tfin, int((tfin-tconv)/dt_out)), method = "nearest")
    N = len(ds.x)
    Nl = len(ds.level)
    Nt = len(ds.time)
    
    time = ds['time']
    
    if chunkdim == 'time':
        
        N_chunk_time = int(N_chunk/(N**2*Nl))
        
        psi = ds['psi'].chunk({'time': N_chunk_time})
        q = ds['q'].chunk({'time': N_chunk_time})
    elif chunkdim == 'space':
        
        N_chunk_space = int(N_chunk/(Nt*Nl*N))
        
        psi = ds['psi'].chunk({'y': N_chunk_space})
        q = ds['q'].chunk({'y': N_chunk_space})
    else:
        psi = ds['psi']
        q = ds['q']
        
    return psi, q, time













