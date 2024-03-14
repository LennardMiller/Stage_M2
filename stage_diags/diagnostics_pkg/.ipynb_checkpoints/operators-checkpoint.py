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


def lap(q, delta):
    ''' computes laplacian of a snapshot field in the interior '''
    
    c = 0
    if len(np.shape(q))==2:
        q = q[None, :, :]
        c = 1
        
    lap_q = (q[:,2:, 1:-1] + q[:,1:-1, 2:] + q[:,:-2, 1:-1] + q[:,1:-1, :-2] - 4*q[:,1:-1, 1:-1])/delta**2
    
    if c == 1:
        lap_q = lap_q[0,:,:]
        
    return lap_q

def lap_x(q):
    ''' computes laplacian of a snapshot field in the interior for xarrays'''
    
    lap_q = q.differentiate('x').differentiate('x') + q.differentiate('y').differentiate('y')
    
    return lap_q

def J(psi, q, delta):
    ''' computes Jacobian between psi and q in the interior using the Arakawa
    method using vectorisation. '''
    
    cpsi = 0
    if len(np.shape(psi))==2:
        psi = psi[None, :, :]
        cpsi = 1
    
    cq = 0
    if len(np.shape(q))==2:
        q = q[None, :, :]
        cq = 1
    
    
    # compute Arakawa jacobian in the interior
    J = (psi[:,2:,1:-1] - psi[:,:-2,1:-1])*(q[:,1:-1,2:] - q[:,1:-1,:-2]) - (psi[:,1:-1,2:] - psi[:,1:-1,:-2])*(q[:,2:,1:-1] - q[:,:-2,1:-1]) # J++
    J += psi[:,2:,1:-1]*(q[:,2:,2:] - q[:,2:,:-2]) - psi[:,:-2,1:-1]*(q[:,:-2,2:] - q[:,:-2, :-2]) - psi[:,1:-1,2:]*(q[:,2:,2:] - q[:,:-2, 2:]) + psi[:,1:-1,:-2]*(q[:,2:,:-2] - q[:,:-2, :-2]) #J+x
    J += -q[:,2:,1:-1]*(psi[:,2:,2:] -psi[:,2:,:-2]) + q[:,:-2,1:-1]*(psi[:,:-2,2:] - psi[:,:-2, :-2]) + q[:,1:-1,2:]*(psi[:,2:,2:] - psi[:,:-2, 2:]) - q[:,1:-1,:-2]*(psi[:,2:,:-2] - psi[:,:-2, :-2]) #Jx+
            
    J /= -12*delta**2 # added the minus for the y/x change when working with x-arrays.
    
    if cpsi == 1 and cq == 1:
        J = J[0,:,:]
    
    return J

def J_x(psi, q):
    ''' computes Jacobian between psi and q in the interior using the Arakawa
    method using vectorisation. Made for xarrays'''
    
    # compute Arakawa jacobian in the interior
    J = (psi.differentiate('x'))*(q.differentiate('y')) - (psi.differentiate('y'))*(q.differentiate('x')) # J++
    J += (psi*q.differentiate('y')).differentiate('x') - (psi*q.differentiate('x')).differentiate('y') #J+x
    J += -(q*psi.differentiate('y')).differentiate('x') + (q*psi.differentiate('x')).differentiate('y') #Jx+
            
    J /= 3
    
    return J

    
def palinstrophy(q, delta):
    ''' calculates the Palinstrophy of a given vorticity field'''
    
    # interior
    
    dqdy = (q[2:,1:-1] - q[0:-2, 1:-1])/(2*delta) 
    dqdx = (q[1:-1,2:] - q[1:-1, 0:-2])/(2*delta) 
    Palinstrophy = np.sum(np.sum((dqdx**2 + dqdy**2)))
    
    # edges
    
    dqdx_west = (-3*q[:,0]  + 4*q[:,1] - q[:,2])/(2*delta)
    dqdx_east = (3*q[:,-1]  - 4*q[:,-2] + q[:,-3])/(2*delta)
    dqdx_north = (q[-1,2:] - q[-1,:-2])/(2*delta)
    dqdx_south = (q[0,2:] - q[0,:-2])/(2*delta)
    

    dqdy_south = (-3*q[0,:]  + 4*q[1,:] - q[2,:])/(2*delta)
    dqdy_north = (3*q[-1,:]  - 4*q[-2,:] + q[-3,:])/(2*delta)
    dqdy_east = (q[2:,-1] - q[:-2,-1])/(2*delta)
    dqdy_west = (q[2:,0] - q[:-2,0])/(2*delta)
    
    # add half sized field on boundaries
    
    Palinstrophy += np.sum(dqdx_west[1:-1]**2/2)
    Palinstrophy += np.sum(dqdx_east[1:-1]**2/2)
    Palinstrophy += np.sum(dqdx_north**2/2)
    Palinstrophy += np.sum(dqdx_south**2/2)
    
    Palinstrophy += np.sum(dqdy_south[1:-1]**2/2)
    Palinstrophy += np.sum(dqdy_north[1:-1]**2/2)
    Palinstrophy += np.sum(dqdy_east**2/2)
    Palinstrophy += np.sum(dqdy_west**2/2)
    
    # add quarter size fields in edges
    
    Palinstrophy += (dqdx_west[0]**2 + dqdy_south[0]**2)/4
    Palinstrophy += (dqdx_west[-1]**2 + dqdy_north[0]**2)/4
    Palinstrophy += (dqdx_east[0]**2 + dqdy_south[-1]**2)/4
    Palinstrophy += (dqdx_east[-1]**2 + dqdy_north[-1]**2)/4
    
    Palinstrophy *= delta**2
    
    return Palinstrophy

def enstrophy_bound(q, delta):
    ''' calculates the enstrophy flux through the boundary for the Enstrophy equation in no slip'''
    
    dqdx_west = (-3*q[:,0]  + 4*q[:,1] - q[:,2])/(2*delta)
    dqdx_east = (3*q[:,-1]  - 4*q[:,-2] + q[:,-3])/(2*delta)
    
    dqdy_south = (-3*q[0,:]  + 4*q[1,:] - q[2,:])/(2*delta)
    dqdy_north = (3*q[-1,:]  - 4*q[-2,:] + q[-3,:])/(2*delta)
    
    # integrate along bounaries
    
    Flux = np.sum(dqdx_west[1:-1]*q[1:-1,0])
    Flux += np.sum(-dqdx_east[1:-1]*q[1:-1,-1])
    Flux += np.sum(dqdy_south[1:-1]*q[0,1:-1])
    Flux += np.sum(-dqdy_north[1:-1]*q[-1,1:-1])
    
    # add corners
    
    Flux += q[0,0]*(dqdx_west[0] + dqdy_south[0])/2
    Flux += q[-1,0]*(dqdx_west[-1] - dqdy_north[0])/2
    Flux += q[0,-1]*(-dqdx_east[0] + dqdy_south[-1])/2
    Flux += q[-1,-1]*(-dqdx_east[-1] - dqdy_north[-1])/2
    
    Flux *= -delta # i calculated the inflow flux, but I want the outflow when integrating from a divergence
    
    return Flux
    
def energy(psi, q, delta):
    ''' calculates total energy of snapshot psi and q '''
    
    Energy = -np.sum(psi*q, axis = (-2, -1))*delta**2/2
    
    return Energy

def enstrophy(q, delta):
    ''' calculates total enstrophy of snapshot q '''
    
    Enstrophy = np.sum(np.sum(q[1:-1, 1:-1]**2/2, axis = -1), axis = -1)
    
    Enstrophy += np.sum(q[0, 1:-1]**2/2, axis = -1)/2
    Enstrophy += np.sum(q[-1, 1:-1]**2/2, axis = -1)/2
    Enstrophy += np.sum(q[1:-1, 0]**2/2, axis = -1)/2
    Enstrophy += np.sum(q[1:-1, -1]**2/2, axis = -1)/2
    
    Enstrophy_ret = Enstrophy*delta**2
    
    return Enstrophy_ret

def circulation(q, delta):
    ''' calculates circulation around the domain '''

    Circulation = np.sum(np.sum(q[1:-1, 1:-1]))
    
    Circulation += np.sum(q[0, 1:-1])/2
    Circulation += np.sum(q[-1, 1:-1])/2
    Circulation += np.sum(q[1:-1, 0])/2
    Circulation += np.sum(q[1:-1, 0])/2
    
    Circulation *= delta**2

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


def calc_uv(psi, delta, bc_fac = 0):
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


def calc_UV(psi, delta, bc_fac = 0):
    ''' calculates u and v for a snapshot'''

    if bc_fac == -1:
        psi = expand_periodic(psi)
    
    N = np.shape(psi)[-1]
    u = np.zeros([N,N])
    u[1:-1,1:-1] = -(psi[2:,1:-1] - psi[:-2,1:-1])/(2*delta)
    
    v = np.zeros([N,N])
    v[1:-1,1:-1] = (psi[1:-1, 2:] - psi[1:-1,:-2])/(2*delta)

    if bc_fac == -1:
        u = u[:,1:-1,1:-1]
        v = v[:,1:-1,1:-1]
        
    return u, v

def calc_ke(psi, L0, N, dh, bc_fac = 0):
    ''' function to calculate energy of simulation. KE is a vector with dimensions m (time) '''
    
    delta = L0/N
    u, v = calc_uv(psi, delta, bc_fac)
    
    KE = integrate(u**2 + v**2, delta, bc_fac)*dh/2
    
    return KE


def calc_pe(psi, L0, N, Ld, dh):
    ''' function to calculate potential energy of multilayered simulation. '''
    
    delta = L0/N
    
    PE = integrate(psi**2*dh/Ld**2, delta)/2
    
    return PE

def calc_pe_diss(psi, nu, L0, N, dh, Ld):
    ''' calculates potential energy dissipation '''
    
    KE = calc_ke(psi,L0,N,dh)
    
    pe_diss = -2*nu*KE/Ld**2
    
    return pe_diss

def calc_ke_diss(psi, nu, L, N, dh, bc_fac):
    ''' calculates kinetic energy dissipation '''
    
    delta = L/N
    
    # calculate ke dissipation
    ke_diss = -nu*dh*integrate(calc_omega(psi, delta, bc_fac)**2, delta, bc_fac)
    
    return ke_diss

def calc_hyper_diss(psi, nu_hyper, L, N, dh, bc_fac):
    """ calculates dissipation due to hyper viscosity with index 6"""

    delta = L/N

    diff = calc_omega(psi, delta, bc_fac)
    diff = calc_omega(diff, delta, bc_fac)
        
    # calculate ke dissipation
    hyper_diss = -nu_hyper*dh*integrate(diff**2, delta, bc_fac)
    
    return hyper_diss

def calc_drag_diss(psi, hEkb, f0, dh, L0, N, bc_fac = 0):

    delta = L0/N

    # calculate dissipation due to drag
    r = hEkb*f0/(2*dh)
    E_kin = calc_ke(psi, L0, N, dh, bc_fac)  
    drag_diss = -2*r*E_kin

    return drag_diss
    
def calc_omega(psi, delta, bc_fac):
    ''' calculates omega '''
    
    c = 0
    if len(np.shape(psi))==2:
        psi = psi[None, :, :]
        c = 1
       
    N = np.shape(psi)[-1]
    Nt = np.shape(psi)[0]
    
    omega = np.zeros([Nt,N,N])
    
    # interior
    omega[:,1:-1,1:-1] = (psi[:,2:, 1:-1] + psi[:,1:-1, 2:] + psi[:,:-2, 1:-1] + psi[:,1:-1, :-2] - 4*psi[:,1:-1, 1:-1])/delta**2
    
    if bc_fac == 1:
        # no-slip boundaries
        omega[:,0,:] = 2*psi[:,1,:]/delta**2
        omega[:,-1,:] = 2*psi[:,-2,:]/delta**2
        omega[:,:,0] = 2*psi[:,:,1]/delta**2
        omega[:,:,-1] = 2*psi[:,:,-2]/delta**2
    
    if bc_fac == 0:
        # free-slip boundaries
        omega[:,0,:] = 0
        omega[:,-1,:] = 0
        omega[:,:,0] = 0
        omega[:,:,-1] = 0

    if bc_fac == -1:
        # periodic boundaries
        omega[:,0,1:-1] = (psi[:,1, 1:-1] + psi[:,0, 2:] + psi[:,-1, 1:-1] + psi[:,0, :-2] - 4*psi[:,0, 1:-1])/delta**2
        omega[:,-1,1:-1] = (psi[:,0, 1:-1] + psi[:,-1, 2:] + psi[:,-2, 1:-1] + psi[:,-1, :-2] - 4*psi[:,-1, 1:-1])/delta**2
        omega[:,1:-1,0] = (psi[:,2:, 0] + psi[:,1:-1, 1] + psi[:,:-2, 0] + psi[:,1:-1, -1] - 4*psi[:,1:-1, 0])/delta**2
        omega[:,1:-1,-1] = (psi[:,2:, -1] + psi[:,1:-1, 0] + psi[:,:-2, -1] + psi[:,1:-1, -2] - 4*psi[:,1:-1, -1])/delta**2

        # corners
        omega[:,0,0] = (psi[:,1, 0] + psi[:,-1,0] + psi[:,0, 1] + psi[:,0,-1] - 4*psi[:,0,0])/delta**2
        omega[:,0,-1] = (psi[:,1, -1] + psi[:,-1,-1] + psi[:,0, 0] + psi[:,0,-2] - 4*psi[:,0,-1])/delta**2
        omega[:,-1,0] = (psi[:,0, 0] + psi[:,-2,0] + psi[:,-1, 1] + psi[:,-1,-1] - 4*psi[:,-1,0])/delta**2
        omega[:,-1,-1] = (psi[:,0, -1] + psi[:,-2,-1] + psi[:,-1, 0] + psi[:,-1,-2] - 4*psi[:,-1,-1])/delta**2
    
    if c == 1:
        omega = omega[0,:,:]
        
    return omega
    
def calc_injection(psi, L, N, tau0, dh):
    ''' calculates energy injection '''
    
    delta = L/N
    
    # create forcing
    x = np.linspace(0, L, int(N) + 1)
    y = x
    xx, yy = np.meshgrid(x, y)
    
    forcing = -tau0/L*2*np.pi*np.sin(2*np.pi*yy.T/L)
    # calculate energy injection
    injection = integrate(-psi*forcing[None, :, :], delta)
    
    return injection


def calc_local_diss(psi, q, delta):
    " calculates the local dissipation field, (del u_i/del x_j)**2"
    
    U, V = calc_UV(psi, delta)
    
    N = np.shape(psi)[-1]
    local_diss = np.zeros([N,N])
    
    local_diss[1:-1, 1:-1] += ((U[1:-1,2:] - U[1:-1,:-2])/(2*delta))**2 # (del u/ del x)**2
    local_diss[1:-1, 1:-1] += ((U[2:,1:-1] - U[:-2, 1:-1])/(2*delta))**2 # (del u/ del y)**2
    local_diss[1:-1, 1:-1] += ((V[1:-1,2:] - V[1:-1,:-2])/(2*delta))**2 # (del v/ del x)**2
    local_diss[1:-1, 1:-1] += ((V[2:,1:-1] - V[:-2, 1:-1])/(2*delta))**2 # (del v/ del y)**2
    
    local_diss[0,:] = q[0,:]
    local_diss[-1,:] = q[-1,:]
    local_diss[:,0] = q[:,0]
    local_diss[:,-1] = q[:,-1]
    
    return local_diss     
    

def omega2_hist(psi, delta, bc_fac, bins):
    """ calculates the histograms at every snapshot, normalised such that np.sum(hist) = 1. """

    omega = calc_omega(psi, delta, bc_fac)
    Nt = np.shape(psi)[0]
    N = np.shape(psi)[-1]
    hists = np.zeros([Nt, len(bins)-1])
    
    for n in range(Nt):
        omega_snap = omega[n,:,:]
        omega_bounds = np.zeros(int((N-2)*4))
        
        for m in range(1, N - 1):
            omega_bounds[4*(m-1)] = omega_snap[0,n]
            omega_bounds[4*(m-1) + 1] = omega_snap[-1,n]
            omega_bounds[4*(m-1) +2] = omega_snap[n,0]
            omega_bounds[4*(m-1) +3] = omega_snap[n,-1]

        omega_snap = omega_snap[1:-1,1:-1].flatten()
            
        hist_interior, bins_ex = np.histogram(omega_snap**2, bins)
        hist_bounds, bins_ex = np.histogram(omega_bounds**2, bins)

        hist_tot = (hist_interior + hist_bounds/2)/(N-1)**2
        hists[n,:] = hist_tot
    
    return hists


def q2_hist(q, bins):
    """ calculates the histograms at every snapshot, normalised such that np.sum(hist) = 1. """

    Nt = np.shape(q)[0]
    N = np.shape(q)[-1]
    hists = np.zeros([Nt, len(bins) - 1])
    
    for n in range(Nt):
        q_snap = q[n,:,:]
        q_bounds = np.zeros(int((N-2)*4))
        
        for m in range(1, N - 1):
            q_bounds[4*(m-1)] = q_snap[0,n]
            q_bounds[4*(m-1) + 1] = q_snap[-1,n]
            q_bounds[4*(m-1) +2] = q_snap[n,0]
            q_bounds[4*(m-1) +3] = q_snap[n,-1]

        q_snap = q_snap[1:-1,1:-1].flatten()
            
        hist_interior, bins_ex = np.histogram(q_snap**2, bins)
        hist_bounds, bins_ex = np.histogram(q_bounds**2, bins)
        
        hist_tot = (hist_interior + hist_bounds/2)/(N-1)**2
        hists[n,:] = hist_tot
    
    return hists