''' collection of functions to apply to layered simulations'''

import numpy as np
import qgutils as qg
from scipy.ndimage import convolve1d
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import dask

def calc_ke(psi, L0, N, dh):
    ''' function to calculate energy of multilayered simulation. KE is a vector with dimensions m (time) x n (layers)'''
    
    delta = L0/N
    u, v = calc_uv(psi, delta)
    
    KE = integrate(u**2 + v**2, delta)*dh/2
    
    return KE

def calc_pe(psi, L0, N, f0, N2, dh):
    ''' function to calculate potential energy of multilayered simulation. '''
    
    delta = L0/N
    
    del_psi = np.array(psi[:,:-1,:,:]) - np.array(psi[:,1:,:,:])
    del_h = (np.array(dh[:-1]) + np.array(dh[1:]))/2
    
    PE = integrate(del_psi**2/del_h[None, :, None, None]*f0**2/N2[None, :, None, None], delta)/2
    
    return PE

def calc_mode_pe(psi, L0, N, f0, N2, dh):
    ''' function that projects psi on the baroclinic modes and calculates potential energy associated to these modes '''
            
    delta = L0/N
    
    Rd, lay2mod, mod2lay = qg.comp_modes(dh, N2, f0, eivec=True, wmode=False, diag=False)
    lam = 1/Rd**2
    lam[0] = 0
    
    psi_m = np.einsum('ij,mjkl->mikl', lay2mod, psi)
    
    u,v = calc_uv(psi_m, delta)
    
    pe = integrate(lam[None, :, None, None]*psi_m**2/2, delta)*np.sum(dh)

    return pe

def calc_mode_ke(psi, L0, N, f0, N2, dh):
    ''' function that projects psi on the baroclinic modes and calculates potential energy associated to these modes '''
            
    delta = L0/N
    
    Rd, lay2mod, mod2lay = qg.comp_modes(dh, N2, f0, eivec=True, wmode=False, diag=False)
    lam = 1/Rd**2
    lam[0] = 0
    
    psi_m = np.einsum('ij,mjkl->mikl', lay2mod, psi)
    
    u,v = calc_uv(psi_m, delta)
    
    ke = integrate((u**2 + v**2)/2, delta)*np.sum(dh)
    
    return ke

def integrate(field, delta):
    ''' integrates field properly on edge-centered grid'''

    c = 0
    if len(np.shape(field))==3:
        field = field[:, None, :, :]
        c = 1
       
    # interior
    integral = np.sum(field[:, :, 1:-1, 1:-1], axis = (-2, -1))
    
    # boundaries
    integral += np.sum(field[:, :, 0, 1:-1], axis = -1)/2
    integral += np.sum(field[:, :, -1, 1:-1], axis = -1)/2
    integral += np.sum(field[:, :, 1:-1, 0], axis = -1)/2
    integral += np.sum(field[:, :, 1:-1, -1], axis = -1)/2
    
    #corners
    integral += field[:, :,0,0]/4
    integral += field[:, :,-1,0]/4
    integral += field[:, :,-1,-1]/4
    integral += field[:, :,0,-1]/4
    
    integral *= delta**2
        
    if c == 1:
        integral = integral[:,0]
        
    return integral


def integrate_3d(field, delta, dh):
    ''' 3D integrates snapshot of field properly on edge-centered grid'''

    # x-y-integrals
    # interior
    integral = np.sum(field[:, 1:-1, 1:-1], axis = (-2, -1))
    
    # boundaries
    integral += np.sum(field[:, 0, 1:-1], axis = -1)/2
    integral += np.sum(field[:, -1, 1:-1], axis = -1)/2
    integral += np.sum(field[:, 1:-1, 0], axis = -1)/2
    integral += np.sum(field[:, 1:-1, -1], axis = -1)/2
    
    #corners
    integral += field[:,0,0]/4
    integral += field[:,-1,0]/4
    integral += field[:,-1,-1]/4
    integral += field[:,0,-1]/4
    
    integral *= delta**2

    integral_3d = np.sum(integral*dh)
    
    return integral_3d

def calc_uv(psi, delta):
    ''' calculates u and v '''
        
    N = np.shape(psi)[-1]
    Nt = np.shape(psi)[0]
    Nl = np.shape(psi)[1]
    
    u = np.zeros([Nt,Nl,N,N])
    u[:,:,1:-1,1:-1] = -(psi[:,:,2:,1:-1] - psi[:,:,:-2,1:-1])/(2*delta)
    
    v = np.zeros([Nt,Nl,N,N])
    v[:,:,1:-1,1:-1] = (psi[:,:,1:-1, 2:] - psi[:,:,1:-1,:-2])/(2*delta)

    return u, v

def calc_omega(psi, delta, bc_fac):
    ''' calculates omega '''
    
    c = 0
    if len(np.shape(psi))==3:
        psi = psi[:, None, :, :]
        c = 1
       
    N = np.shape(psi)[-1]
    Nt = np.shape(psi)[0]
    Nl = np.shape(psi)[1]
    
    omega = np.zeros([Nt,Nl,N,N])
    
    # interior
    omega[:,:,1:-1,1:-1] = (psi[:,:,2:, 1:-1] + psi[:,:,1:-1, 2:] + psi[:,:,:-2, 1:-1] + psi[:,:,1:-1, :-2] - 4*psi[:,:,1:-1, 1:-1])/delta**2
    
    if bc_fac == 1:
        # no-slip boundaries
        omega[:,:,0,:] = 2*psi[:,:,1,:]/delta**2
        omega[:,:,-1,:] = 2*psi[:,:,-2,:]/delta**2
        omega[:,:,:,0] = 2*psi[:,:,:,1]/delta**2
        omega[:,:,:,-1] = 2*psi[:,:,:,-2]/delta**2
    
    if bc_fac == 0:
        # free-slip boundaries
        omega[:,:,0,:] = 0
        omega[:,:,-1,:] = 0
        omega[:,:,:,0] = 0
        omega[:,:,:,-1] = 0
    
    if c == 1:
        omega = omega[:,0,:,:]
        
    return omega

def J(psi, q, delta):
    ''' computes Jacobian between psi and q in the interior using the Arakawa
    method using vectorisation. '''
    
    cpsi = 0
    if len(np.shape(psi))==3:
        psi = psi[None, :, :,:]
        cpsi = 1
    
    cq = 0
    if len(np.shape(q))==3:
        q = q[None, :, :,:]
        cq = 1

    if cpsi == 0:
        J = np.zeros(np.shape(psi))
    else:
        J = np.zeros(np.shape(q))

    # compute Arakawa jacobian in the interior
    J[:,:,1:-1,1:-1] += (psi[:,:,2:,1:-1] - psi[:,:,:-2,1:-1])*(q[:,:,1:-1,2:] - q[:,:,1:-1,:-2]) - (psi[:,:,1:-1,2:] - psi[:,:,1:-1,:-2])*(q[:,:,2:,1:-1] - q[:,:,:-2,1:-1]) # J++
    J[:,:,1:-1,1:-1] += psi[:,:,2:,1:-1]*(q[:,:,2:,2:] - q[:,:,2:,:-2]) - psi[:,:,:-2,1:-1]*(q[:,:,:-2,2:] - q[:,:,:-2, :-2]) - psi[:,:,1:-1,2:]*(q[:,:,2:,2:] - q[:,:,:-2, 2:]) + psi[:,:,1:-1,:-2]*(q[:,:,2:,:-2] - q[:,:,:-2, :-2]) #J+x
    J[:,:,1:-1,1:-1] += -q[:,:,2:,1:-1]*(psi[:,:,2:,2:] -psi[:,:,2:,:-2]) + q[:,:,:-2,1:-1]*(psi[:,:,:-2,2:] - psi[:,:,:-2, :-2]) + q[:,:,1:-1,2:]*(psi[:,:,2:,2:] - psi[:,:,:-2, 2:]) - q[:,:,1:-1,:-2]*(psi[:,:,2:,:-2] - psi[:,:,:-2, :-2]) #Jx+
            
    J /= -12*delta**2 # added the minus for the y/x change when working with x-arrays.
    
    if cpsi == 1 and cq == 1:
        J = J[0,:,:,:]
    
    return J

def calc_baroclinic_transfer(psi, psi_mean, omega_mean, nu, L0, N, f0, tau0, hEkb, N2, dh, bc_fac):

    delta = L0/N
    
    psi_fluc = psi - psi_mean
   
    baroclinic_transfer = 2*f0**2/(N2[0]*np.sum(dh))*(psi_mean[1,:,:] - psi_mean[0,:,:])*J(psi_fluc[:,0,:,:],psi_fluc[:,1,:,:], delta)
    baroclinic_transfer = np.sum(baroclinic_transfer*delta**2, axis = (1,2))
    
    return baroclinic_transfer

def calc_barotropic_transfer(psi, psi_mean, omega_mean, nu, L0, N, f0, tau0, hEkb, N2, dh, bc_fac):
    
    delta = L0/N
    
    psi_fluc = psi - psi_mean
    omega_fluc = calc_omega(psi, delta, bc_fac) - omega_mean
    
    barotropic_transfer = dh[None,:,None,None]*(psi_fluc*J(psi_fluc, omega_mean, delta) + psi_fluc*J(psi_mean, omega_fluc, delta))
    barotropic_transfer = np.sum(barotropic_transfer*delta**2, axis = (1,2,3))
    
    return barotropic_transfer

def calc_injection(psi, nu, L, N, f0, tau0, hEkb, N2, dh, bc_fac):
    ''' calculates energy injection '''
    
    delta = L/N
    
    # create forcing
    x = np.linspace(0, L, int(N) + 1)
    y = x
    xx, yy = np.meshgrid(x, y)
    
    forcing = -tau0/L*2*np.pi*np.sin(2*np.pi*xx.T/L) # should be xx if we work with normal numpy arrays, but yy if we work within parallelized dask arrays
    
    # calculate energy injection
    injection = integrate(-psi[:,0,:,:]*forcing[None, :, :], delta)
    
    return injection

def calc_ke_diss(psi, nu, L, N, f0, tau0, hEkb, N2, dh, bc_fac):
    ''' calculates kinetic energy dissipation '''
    
    delta = L/N
    
    # calculate ke dissipation
    ke_diss = -nu*dh*integrate(calc_omega(psi, delta, bc_fac)**2, delta)
    
    return ke_diss

def calc_pe_diss(psi, nu, L, N, f0, tau0, hEkb, N2, dh, bc_fac):
    ''' calculates potential energy dissipation '''
    
    delta = L/N
    
    # calculate pe dissipation
    del_psi = np.array(psi[:,:-1,:,:]) - np.array(psi[:,1:,:,:])
    del_h = (np.array(dh[:-1]) + np.array(dh[1:]))/2
    
    pe_diss = -nu*integrate(-f0**2/N2[None, :, None, None]/del_h[None, :, None, None]*del_psi*calc_omega(del_psi, delta, bc_fac), delta)
    
    return pe_diss

def calc_bottom_diss(psi, nu, L, N, f0, tau0, hEkb, N2, dh, bc_fac):
    ''' calculates bottom energy dissipation '''
    
    delta = L/N
    
    # calculate bottom dissipation
    bottom_diss = -hEkb*f0/2*integrate(-psi[:,-1,:,:]*calc_omega(psi[:,-1,:,:], delta, bc_fac), delta)
    
    return bottom_diss

def gaussian_filter(field, time, omega):
    ''' computes gaussian filtered version of field at frequency omega '''

    #create filter envelope on same grid as field variable
    dt = time[1] - time[0]
    T = 1/omega
    dt_per_T = int(T/dt)
    N_T = 3
    
    t_filter = np.linspace(-N_T*dt_per_T*dt, N_T*dt_per_T*dt, 2*N_T*dt_per_T+1)
    
    filter = omega/(np.sqrt(2*np.pi))*np.exp(-1/2*(omega*t_filter)**2)

    # uniform_filter1d for top hat kernel
    filtered_field = convolve1d(field, filter, axis = -1, mode = 'constant', origin = 0)*dt # core dimension of operation is set towards the end
    filtered_field = filtered_field[:,:,:,N_T*dt_per_T:-N_T*dt_per_T]
    
    return filtered_field

def top_hat_filter(field, time, omega):
    ''' computes gaussian filtered version of field at frequency omega '''

    #create filter envelope on same grid as field variable
    dt = time[1] - time[0]
    T = 1/omega
    dt_per_T = int(T/dt)+1
    
    # uniform_filter1d for top hat kernel
    filtered_field = uniform_filter1d(field, dt_per_T, axis = 0)
    filtered_field = filtered_field[int(dt_per_T/2):-int(dt_per_T/2),:,:,:]
    
    return filtered_field

@dask.delayed()
def grained_psiq(psi, q, time, omega):
    ''' calculates cumulative coarse grained product of psi and q at frequency omega. '''
    
    psi_filt = top_hat_filter(psi, time, omega)
    q_filt = top_hat_filter(q, time, omega)
    psi_q_filt = np.mean(-psi_filt*q_filt, axis = 0)
    
    return psi_q_filt


def correlation_histogram(q, bins):
    """ calculates the joint PDFs of log(q1**2) and log(q2**2), at each snapshot """
    
    Nt = np.shape(q)[0]
    N = np.shape(q)[1]  # space will be broadcast to last dimension
    
    Hs = np.zeros([Nt, len(bins), len(bins)])
    
    for n in range(Nt):
        q1_snap = q[n,:,:,0].flatten()
        q2_snap = q[n,:,:,1].flatten()
        
        H, xedges, yedges = np.histogram2d(q1_snap, q2_snap, bins=[bins, bins], density=True)
        Hs[n,:-1,:-1] = H.T

    return Hs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    