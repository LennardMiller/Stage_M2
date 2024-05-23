""" Script to calculate physical parameters corresponding to non-dimensional 
numbers Oc and Zo. """

import numpy as np
import matplotlib.pyplot as plt

#Pour check la simu 83:
Oc = 0.017
Zo = 2.8



def dim_vars(Oc, Zo):
    """ calculates dimensional variables for a given Oc and Zo"""
    
    factor_k_H_Z = 5*np.pi*2#2 # sets the distance between k_f and k_H_Z
    factor_k_eps = 0.86*np.pi*2#1/2 # sets the distance between k_f and k_eps
    
    n_hyper = 6
    
    # two physical units (meters and seconds), i.e. we are allowed to set two
    # dimensional variables to 1. I choose k_f and epsilon.
    
    k_f = 32
    eps = 4e-6
    
    #setting the enstrophy dissipation scale sets the viscosity
    k_H_Z = factor_k_H_Z*k_f
    nu = (eps*k_f**2)**(1/3)/((k_H_Z)**n_hyper)
    
    # setting k_eps fixes beta
    k_eps = factor_k_eps*k_f
    beta = ((k_eps)**5*eps)**(1/3)
    
    # setting zonostrophy sets the drag
    k_Rh = k_eps/Zo
    r = eps/beta**2*(k_Rh)**4
    
    # setting occupation sets the domain length
    k_L = Oc*k_Rh
    L = 2*np.pi/k_L/((2*np.pi))
    
    # the numerical resolution is set by the ratio between L and the length scale 
    # corresponding to k_H_Z (I added a factor of 2 to be sure, maybe needs to be
    # changed)
    
    N = 2*L*(k_H_Z)
    
    return k_f, eps, nu, beta, r, L, N

# sample calculation

k_f, eps, nu, beta, r, L, N = dim_vars(Oc, Zo)

print(f"k_f = {k_f}")
print(f"sigma_f = {np.sqrt(eps)}")
print(f"nu_hyper = {nu}")
print(f"beta = {beta}")
print(f"r = {r}")
print(f"L = {L}")
print(f"N = {N}")

# check if everything worked fine (also here you see my definitions, 
# I included factors of 2pi everywhere)

n_hyper = 6

Oc_test = (eps/(beta**2*r*L**4))**(1/4)
Zo_test = (beta**2*eps/r**5)**(1/20)

k_Rh = 2*np.pi*(beta**2*r/eps)**(1/4)
k_eps = 2*np.pi*(beta**3/eps)**(1/5)
k_L = 2*np.pi/L
k_H_Z = 2*np.pi*((eps*k_f**2)**(1/3)/nu)**(1/n_hyper)


# plot scales
plt.title(f"Scale comparison for Oc = {Oc}, Zo = {Zo}")
plt.scatter(k_f, 1, label = "k_f")
plt.scatter(k_H_Z, 1, label = "k_H_Z")
plt.scatter(k_eps, 1, label = "k_epsilon")
plt.scatter(k_Rh, 1, label = "k_Rh")
plt.scatter(k_L, 1, label = "k_L")
plt.gca().set_xscale('log')
plt.xlabel("k")
plt.legend()
plt.show()

# create plot to see resolution on the parameter space

Ocs = np.logspace(-1, 1, 100)
Zos = np.logspace(0, 1, 100)

OCs, ZOs = np.meshgrid(Ocs, Zos)

Resolutions = dim_vars(OCs, ZOs)

plt.pcolormesh(Zos, Ocs, np.log10(Resolutions[-1]).T, vmin = 0, vmax = 3)
plt.colorbar(label = r"$log_{10}(N)$")
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.title("Required spacial resolution")
plt.xlabel("Zo")
plt.ylabel("Oc")
plt.show()


# create plot to see time steps for spinup on the parameter space

Resolutions = dim_vars(OCs, ZOs)

N_t = Resolutions[-1]*OCs*ZOs**5

plt.pcolormesh(Zos, Ocs, np.log10(N_t.T))
plt.colorbar(label = r"$log_{10}(N_t)$")
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.title("Required time steps for convergence")
plt.xlabel("Zo")
plt.ylabel("Oc")
plt.show()


# create plot to measure time that simulation should take

T_tot = N_t*Resolutions[-1]

plt.pcolormesh(Zos, Ocs, np.log10(T_tot.T))
plt.colorbar(label = r"$log_{10}(T_{tot})$")
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.title("Cluster time required for convergence")
plt.xlabel("Zo")
plt.ylabel("Oc")
plt.show()













