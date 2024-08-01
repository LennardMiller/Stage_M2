""" Script to calculate physical parameters corresponding to non-dimensional 
numbers Oc and Zo. """

import numpy as np
import matplotlib.pyplot as plt


Oc = 0.1
Zo = 8

def dim_vars(Oc, Zo):
    """ calculates dimensional variables for a given Oc and Zo"""
    
    factor_k_H_Z = 6 # sets the distance between k_f and k_H_Z
    factor_k_eps = 1 # sets the distance between k_f and k_eps
    
    n_hyper = 4
    
    # two physical units (meters and seconds), i.e. we are allowed to set two
    # dimensional variables to 1. I choose k_f and epsilon.
    
    k_f = 1
    eps = 1
    
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
    L = 2*np.pi/k_L
    
    # the numerical resolution is set by the ratio between L and the length scale 
    # corresponding to k_H_Z (I added a factor of 2 to be sure, maybe needs to be
    # changed)
    
    N = L/(2*np.pi)*(k_H_Z)*4
    
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


# check if everything worked fine 

n_hyper = 4

Oc_test = (eps/(beta**2*r*(L/(2*np.pi))**4))**(1/4) #k_Rh/k_L
Zo_test = (beta**2*eps/r**5)**(1/20) #k_Rh/k_eps
print("Oc test",Oc_test)
print("Zo test",Zo_test)
print
k_Rh = (beta**2*r/eps)**(1/4)
k_eps = (beta**3/eps)**(1/5)
k_L = 2*np.pi/L
k_H_Z = ((eps*k_f**2)**(1/3)/nu)**(1/n_hyper)

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













