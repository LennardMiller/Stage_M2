
import numpy as np
import sys
sys.path.append('/home/massoale/Stage_M2/Analyse/qgutils-master/')
sys.path.append('/home/massoale/Bureau/Stage_M2/stage_diags/diagnostics_pkg/')
import io_utils as io
import netCDF4 as nc





#Number of simulation
n=48

#choose between 'local' or 'dahu'
where='dahu'


#Reading the netcdf file
if where=='local':
    if n<10:
        simu_name='outdir_000'+str(n)
    elif n<100 and n>=10:
        simu_name='outdir_00'+str(n)
    Path='/home/massoale/Simu_Test/qgw-main/src/'+simu_name+'/'

elif where=='dahu':
    simu_name='dahu_'+str(n)
    Path='/home/massoale/Simu_Test/simu_dahu/simu_dahu'+str(n)+'/outdir_0001/'

elif where=='dahu_job':
    simu_name='dahu_'+str(n)
    Path='simu_dahu'+str(n)+'/outdir_0001/'
else:
    print('Error: where not recognized')
    sys.exit()
print('la simulation chargée est: ' + simu_name )
print("depuis: "+where)

filenames=['/vars.nc']



dataset=nc.Dataset(Path+filenames[0])
#dataset=nc.Dataset('./wave_simu_stocka0.nc/outdir_0001/vars.nc')



t=dataset.variables['time'][350:]
x=dataset.variables['x'][350:]
y=dataset.variables['y'][:]
psi=dataset.variables['psi'][350:,0,:,:]
q=dataset.variables['q'][350:,0,:,:]



param=io.read_params(Path)
print(param)
f0= param['f0']
beta=param['beta']
hEkb=param['hEkb']
dh=param['dh'][0]
Lx=param['Lx']
nx=param['NX']
ny=param['NY']
k_f=param['k_f']
sigma_f=param['sigma_f']
bc_fac=param['bc_fac']
nu_hyper=param['nu_hyper']
n_hyper=param['n_hyper']
dt_out=param['dt_out']


def fft2d_RI(psi, Lx, nx, ny,time_sel=1):
    
    dx = Lx / nx
    dy = Lx / ny

    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi  
    
    kx_shifted = np.fft.fftshift(kx)
    ky_shifted = np.fft.fftshift(ky)
    
    Kx, Ky = np.meshgrid(kx_shifted, ky_shifted)
    #Ky = np.flipud(Ky)

    
    psi_data = psi[time_sel, :, :]

    # 2D Fourier Transform
    fft_result = np.fft.fft2(psi_data) / (nx * ny) 
    fft_shifted = np.fft.fftshift(fft_result)  # Shift zero frequency component to center


    return fft_shifted, Kx, Ky


def fft_kpluk0(fft,k,l,bool_plus=True,L=Lx):
    """
    fft: FFT de psi
    k,l: Wavenumbers to shift (k0+k,l0+l)

    Calcul psi(K+K0) en vectorisant. 
    Retourne une matrice de même taille que fft avec les valeurs de psi(K+K0) shiftées
    
    Si Bool_plus=false calcule psi(K-K0) shiftée
    """

        
    New_fft_shift=np.zeros_like(fft)

    #Pour le calcul de FFT de (k-k0,l-l0)
    if bool_plus == False:
        k=-k
        l=-l

    m=int(np.round(L*k/(2*np.pi)))
    n=int(np.round(L*l/(2*np.pi)))
    
    
    if k >= 0 and l >= 0:

        if n==0 and m!=0:
            Mat_fft_cut=fft[:,m:]
            New_fft_shift[:,:-m]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[n:,:]
            New_fft_shift[:-n,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[n:,m:]
            New_fft_shift[:-n,:-m]=Mat_fft_cut

    elif k < 0 and l < 0:

        if n==0 and m!=0:
            Mat_fft_cut=fft[:,:m]
            New_fft_shift[:,-m:]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[:n,:]
            New_fft_shift[-n:,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[:n,:m]
            New_fft_shift[-n:,-m:]=Mat_fft_cut

    elif k >= 0 and l < 0:

        if n==0 and m!=0:
            Mat_fft_cut=fft[:,m:]
            New_fft_shift[:,:-m]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[:n,:]
            New_fft_shift[-n:,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[:n,m:]
            New_fft_shift[-n:,:-m]=Mat_fft_cut

    elif k < 0 and l >= 0:


        if n==0 and m!=0:
            Mat_fft_cut=fft[:,:m]
            New_fft_shift[:,-m:]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[n:,:]
            New_fft_shift[:-n,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[n:,:m]
            New_fft_shift[:-n,-m:]=Mat_fft_cut

    return New_fft_shift

def fft_kpluk0(fft,k,l,bool_plus=True,L=Lx):
    """
    fft: FFT de psi
    k,l: Wavenumbers to shift (k0+k,l0+l)

    Calcul psi(K+K0) en vectorisant. 
    Retourne une matrice de même taille que fft avec les valeurs de psi(K+K0) shiftées
    
    Si Bool_plus=false calcule psi(K-K0) shiftée
    """

        
    New_fft_shift=np.zeros_like(fft)

    #Pour le calcul de FFT de (k-k0,l-l0)
    if bool_plus == False:
        k=-k
        l=-l

    m=int(np.round(L*k/(2*np.pi)))
    n=int(np.round(L*l/(2*np.pi)))
    
    
    if k >= 0 and l >= 0:

        if n==0 and m!=0:
            Mat_fft_cut=fft[:,m:]
            New_fft_shift[:,:-m]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[n:,:]
            New_fft_shift[:-n,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[n:,m:]
            New_fft_shift[:-n,:-m]=Mat_fft_cut

    elif k < 0 and l < 0:

        if n==0 and m!=0:
            Mat_fft_cut=fft[:,:m]
            New_fft_shift[:,-m:]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[:n,:]
            New_fft_shift[-n:,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[:n,:m]
            New_fft_shift[-n:,-m:]=Mat_fft_cut

    elif k >= 0 and l < 0:

        if n==0 and m!=0:
            Mat_fft_cut=fft[:,m:]
            New_fft_shift[:,:-m]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[:n,:]
            New_fft_shift[-n:,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[:n,m:]
            New_fft_shift[-n:,:-m]=Mat_fft_cut

    elif k < 0 and l >= 0:


        if n==0 and m!=0:
            Mat_fft_cut=fft[:,:m]
            New_fft_shift[:,-m:]=Mat_fft_cut

        elif n!=0 and m==0:
            Mat_fft_cut=fft[n:,:]
            New_fft_shift[:-n,:]=Mat_fft_cut

        elif n==0 and m==0:
            Mat_fft_cut=fft[:,:]
            New_fft_shift[:,:]=Mat_fft_cut

        else:
            Mat_fft_cut=fft[n:,:m]
            New_fft_shift[:-n,-m:]=Mat_fft_cut

    return New_fft_shift

#Computing non linear energy, with vectorisation

def non_linear_energy_vect(fft,Kx,Ky,L=Lx):
    """
    fft: 2D array of the fft of psi
    Kx,Ky: 2D arrays of the wavenumbers

    return the non linear energy term in the Fourier space
    """

    NL_energy=np.zeros_like(fft)

    for i,l in enumerate(Ky[:,0]):
        for j,k in enumerate(Kx[0,:]):
            #Finding the indices of k,l
            m=int(np.round(L*k/(2*np.pi)))+len(Kx[0,:])//2
            n=int(np.round(L*l/(2*np.pi)))+len(Ky[:,0])//2


            NL_energy += 1/4*(k*Ky-l*Kx) * (\
                            ((k-Kx)**2+(l-Ky)**2)*( fft[n,m]*fft_kpluk0(fft,k,l,False,L)*np.conj(fft)\
                                                               + np.conj(fft[n,m])*np.conj(fft_kpluk0(fft,k,l,False,L))*fft )\
                            -  ((Kx+k)**2+(Ky+l)**2)*( np.conj(fft[n,m])*fft_kpluk0(fft,k,l,True,L)*np.conj(fft)\
                                                            + fft[n,m]*np.conj(fft_kpluk0(fft,k,l,True,L))*fft )\
                            )
        print(f"Processing row {i+1}/{len(Ky[:, 0])}")

    return NL_energy


 

from joblib import Parallel, delayed



def parallel_non_linear_energy(ffts, Kx, Ky, L=Lx, n_jobs=-1):
    """
    ffts: List of 2D arrays of the fft of psi over time
    Kx, Ky: 2D arrays of the wavenumbers
    L: Length scale
    n_jobs: Number of jobs for parallel processing

    Returns a list of non-linear energy terms for each time snapshot
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(non_linear_energy_vect)(fft, Kx, Ky, L) for fft in ffts
    )
    return results


nx = int(nx)
ny = int(ny)
time_idx=np.arange(0,8,1)
ffts = [fft2d_RI(psi, Lx, nx, ny, time_sel=i)[0] for i in time_idx]
Kx,Ky = fft2d_RI(psi, Lx, nx, ny, time_sel=0)[1:3]
non_linear_energies = parallel_non_linear_energy(ffts, Kx, Ky, Lx)

np.save('non_linear_energy_matrix/non_lin_term_list.npy', non_linear_energies)

