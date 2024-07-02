''' groups useful functions to calculate operators with node-centered fields (size 
= N + 1).'''

import numpy as np
import xarray as xr 

def integrate(field, delta, bc_fac = 0):
    ''' integrates a field including the fact that the cells on the boundary are only half cells '''
    
    c = 0
    if len(np.shape(field))==2:
        field = field[None, :, :]
        c = 1

    if bc_fac != -1:
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
    else:
        # interior
        integral = np.sum(field, axis = (-2, -1))
        
    integral *= delta**2
    
    if c == 1:
        integral = integral[0]
        
    return integral


def J(psi, q, delta, bc_fac = 0):
    ''' computes Jacobian between psi and q in the interior using the Arakawa
    method. '''
    
    cpsi = 0
    if len(np.shape(psi))==2:
        psi = psi[None, :, :]
        cpsi = 1
    
    cq = 0
    if len(np.shape(q))==2:
        q = q[None, :, :]
        cq = 1

    if bc_fac == -1:
        psi = expand_periodic(psi)
        q = expand_periodic(q)
    
    # compute Arakawa jacobian in the interior
    J = (psi[:,2:,1:-1] - psi[:,:-2,1:-1])*(q[:,1:-1,2:] - q[:,1:-1,:-2]) - (psi[:,1:-1,2:] - psi[:,1:-1,:-2])*(q[:,2:,1:-1] - q[:,:-2,1:-1]) # J++
    J += psi[:,2:,1:-1]*(q[:,2:,2:] - q[:,2:,:-2]) - psi[:,:-2,1:-1]*(q[:,:-2,2:] - q[:,:-2, :-2]) - psi[:,1:-1,2:]*(q[:,2:,2:] - q[:,:-2, 2:]) + psi[:,1:-1,:-2]*(q[:,2:,:-2] - q[:,:-2, :-2]) #J+x
    J += -q[:,2:,1:-1]*(psi[:,2:,2:] -psi[:,2:,:-2]) + q[:,:-2,1:-1]*(psi[:,:-2,2:] - psi[:,:-2, :-2]) + q[:,1:-1,2:]*(psi[:,2:,2:] - psi[:,:-2, 2:]) - q[:,1:-1,:-2]*(psi[:,2:,:-2] - psi[:,:-2, :-2]) #Jx+
            
    J /= -12*delta**2 # added the minus for the y/x change when working with x-arrays.

    if bc_fac == -1:
        J[:,1:-1,1:-1]
        
    if cpsi == 1 and cq == 1:
        J = J[0,:,:]
    
    return J
    
def expand_periodic(arr):
    ''' pads psi with periodic values on boundaries '''
    
    N = np.shape(arr)[-1]
    Nt = np.shape(arr)[0]
    N += 2
    arr_new = np.zeros([Nt,N,N])

    # sides
    arr_new[:,1:-1,1:-1] = arr
    arr_new[:,0,1:-1] = arr[:,-1,:]
    arr_new[:,-1,1:-1] = arr[:,0,:]
    arr_new[:,1:-1,0] = arr[:,:,-1]
    arr_new[:,1:-1,-1] = arr[:,:,0]

    # corners
    arr_new[:,0,0] = arr[:,-1,-1]
    arr_new[:,-1,0] = arr[:,0,-1]
    arr_new[:,0,-1] = arr[:,-1,0]
    arr_new[:,-1,-1] = arr[:,0,0]
    
    return arr_new
    
def calc_en(psi, q, delta, bc_fac):
    ''' calculates total energy of snapshot psi and q '''
    
    Energy = -integrate(psi*q/2, delta, bc_fac)
    
    return Energy


def calc_uv(psi, delta, bc_fac):
    ''' calculates u and v '''

    if bc_fac == -1:
        psi = expand_periodic(psi)
    
    N = np.shape(psi)[-1]
    Nt = np.shape(psi)[0]
    
    u = np.zeros([Nt,N,N])
    u[:,1:-1,1:-1] = -(psi[:,2:,1:-1] - psi[:,:-2,1:-1])/(2*delta)
    
    v = np.zeros([Nt,N,N])
    v[:,1:-1,1:-1] = (psi[:,1:-1, 2:] - psi[:,1:-1,:-2])/(2*delta)

    if bc_fac == -1:
        u = u[:,1:-1,1:-1]
        v = v[:,1:-1,1:-1]
        
    return u, v

def calc_hyper_diss(psi, nu_hyper, delta, bc_fac):
    """ calculates dissipation due to hyper viscosity with index 6"""

    diff = calc_omega(psi, delta, bc_fac)
    diff = calc_omega(diff, delta, bc_fac)
        
    # calculate ke dissipation
    hyper_diss = -nu_hyper*integrate(diff**2, delta, bc_fac)
    
    return hyper_diss

def calc_drag_diss(psi, hEkb, f0, delta, bc_fac = 0):

    # calculate dissipation due to drag
    r = hEkb*f0/2
    E_kin = calc_ke(psi, q, delta, bc_fac)  
    drag_diss = -2*r*E_kin

    return drag_diss
    
def calc_lap(psi, delta, bc_fac):
    ''' calculates laplacian  of field psi'''
    
    c = 0
    if len(np.shape(psi))==2:
        psi = psi[None, :, :]
        c = 1
       
    N = np.shape(psi)[-1]
    Nt = np.shape(psi)[0]
    print("Nt=",Nt,"N=",N)
    lap = np.zeros([Nt,N,N])
    print("lap de ope shape :",lap.shape)
    print("psi de op shape",psi.shape)
    print("1",psi[:,2:, 1:-1].shape)
    print("2",psi[:,1:-1, 2:].shape)
    print("3",psi[:,:-2, 1:-1].shape)
    print("4",psi[:,1:-1, :-2].shape)
    print("5",psi[:,1:-1, 1:-1].shape)
    print("-1", lap[:,1:-1,1:-1].shape)
    print(delta.shape)
    # interior
    lap[:,1:-1,1:-1] = (psi[:,2:, 1:-1] + psi[:,1:-1, 2:] + psi[:,:-2, 1:-1] + psi[:,1:-1, :-2] - 4*psi[:,1:-1, 1:-1])/delta**2
    #lap[:,:,:] = (psi[:,2:, 1:-1] + psi[:,1:-1, 2:] + psi[:,:-2, 1:-1] + psi[:,1:-1, :-2] - 4*psi[:,1:-1, 1:-1])/delta**2
    if bc_fac == 1:
        # no-slip boundaries
        lap[:,0,:] = 2*psi[:,1,:]/delta**2
        lap[:,-1,:] = 2*psi[:,-2,:]/delta**2
        lap[:,:,0] = 2*psi[:,:,1]/delta**2
        lap[:,:,-1] = 2*psi[:,:,-2]/delta**2
    
    if bc_fac == 0:
        # free-slip boundaries
        lap[:,0,:] = 0
        lap[:,-1,:] = 0
        lap[:,:,0] = 0
        lap[:,:,-1] = 0

    if bc_fac == -1:
        # periodic boundaries
        lap[:,0,1:-1] = (psi[:,1, 1:-1] + psi[:,0, 2:] + psi[:,-1, 1:-1] + psi[:,0, :-2] - 4*psi[:,0, 1:-1])/delta**2
        lap[:,-1,1:-1] = (psi[:,0, 1:-1] + psi[:,-1, 2:] + psi[:,-2, 1:-1] + psi[:,-1, :-2] - 4*psi[:,-1, 1:-1])/delta**2
        lap[:,1:-1,0] = (psi[:,2:, 0] + psi[:,1:-1, 1] + psi[:,:-2, 0] + psi[:,1:-1, -1] - 4*psi[:,1:-1, 0])/delta**2
        lap[:,1:-1,-1] = (psi[:,2:, -1] + psi[:,1:-1, 0] + psi[:,:-2, -1] + psi[:,1:-1, -2] - 4*psi[:,1:-1, -1])/delta**2

        # corners
        lap[:,0,0] = (psi[:,1, 0] + psi[:,-1,0] + psi[:,0, 1] + psi[:,0,-1] - 4*psi[:,0,0])/delta**2
        lap[:,0,-1] = (psi[:,1, -1] + psi[:,-1,-1] + psi[:,0, 0] + psi[:,0,-2] - 4*psi[:,0,-1])/delta**2
        lap[:,-1,0] = (psi[:,0, 0] + psi[:,-2,0] + psi[:,-1, 1] + psi[:,-1,-1] - 4*psi[:,-1,0])/delta**2
        lap[:,-1,-1] = (psi[:,0, -1] + psi[:,-2,-1] + psi[:,-1, 0] + psi[:,-1,-2] - 4*psi[:,-1,-1])/delta**2
    
    if c == 1:
        lap = lap[0,:,:]
        
    return lap
