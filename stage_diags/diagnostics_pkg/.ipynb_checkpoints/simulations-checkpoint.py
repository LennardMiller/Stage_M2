import numpy as np
import matplotlib.pyplot as plt
from diagnostics_pkg.simulation import simulation
from diagnostics_pkg import operators

class simulations:
        ''' Class to hold multiple simulations and to plot diagnostics as 
        functions of non-dimensional variables'''

        def __init__(self, Paths, Iconvs):
                self.paths = Paths
                self.iconvs = Iconvs
                self.simulations = [simulation(path, Read_q = 1) for path in self.paths]
                
        def plt_dia_conv(self):
                ''' plots the diagnostics as a function of N to show convergence of all simulations'''
        
                for n, sim in enumerate(self.simulations):
                        sim.read_path(Iconv = self.iconvs[n])
                        sim.calc_ke(autocorr = 1, plot = 0)
                        sim.calc_diss(autocorr = 1, plot = 0)
                        sim.calc_input(autocorr = 1, plot = 0)
                        sim.drop_psi()
        
                Ns = [sim.N for sim in self.simulations]
                
                disss = [sim.diss_mean for sim in self.simulations]
                diss_errs = [sim.diss_mean_err for sim in self.simulations]
                
                kes = [sim.ke_mean for sim in self.simulations]
                ke_errs = [sim.ke_mean_err for sim in self.simulations]
                
                inputs = [sim.input_mean for sim in self.simulations]
                input_errs = [sim.input_mean_err for sim in self.simulations]
                
                plt.errorbar(Ns, kes, yerr = ke_errs)
                plt.xscale("log")
                plt.xlabel('N')
                plt.ylabel('Kinetic Energy mean')
                plt.title('Convergence of energy at Re = 283')
                plt.show()
                
                plt.errorbar(Ns, disss, yerr = diss_errs)
                plt.xscale("log")
                plt.xlabel('N')
                plt.ylabel('Dissipation mean')
                plt.title('Convergence of dissipation at Re = 283')
                plt.show()
                
                plt.errorbar(Ns, inputs, yerr = input_errs)
                plt.xscale("log")
                plt.xlabel('N')
                plt.ylabel('input mean')
                plt.title('Convergence of energy input at Re = 283')
                plt.show()

        def plt_spec_conv(self):
            
            ''' Plots spectral convergence, i.e. the spectra for different N (or Re)'''
            
            colors = [['g-', 'g--'], ['b-', 'b--'], ['r-', 'r--'], ['k-', 'k--'], ['y-', 'y--']]
            
            for n, simulation in enumerate(self.simulations):
                    simulation.read_path(sizelim = 10e7, Iconv = self.iconvs[n])
                    simulation.calc_spec(window = 0)
                    plt.loglog(simulation.kr, simulation.spec, colors[n][0], label = f'Re = {simulation.Re:.0f}')
                    
                    # calculate spectral gradients
                    # log_spec = np.log(simulation.spec)
                    # log_kr = np.log(simulation.kr)
                    # N_k = len(simulation.kr)
                    # Boundfit = list(range(int(N_k/8), int(N_k/4)))
                    # log_spec_fit = log_spec[Boundfit]
                    # log_k_fit = log_kr[Boundfit]
                    # z = np.polyfit(log_k_fit, log_spec_fit, 1)
                    # print(f'spectral gradient was found to be {z[0]} at N = {simulation.N}')
                    # k_fit = simulation.kr[Boundfit]
                    # fit = np.exp(z[1])*k_fit**z[0]
                    #plt.loglog(k_fit, fit)
                    
                    # calculate and plot Rhines scale
                    Urms = np.sqrt(np.sum(simulation.spec)*2*np.pi/simulation.L)
                    k_rhines = np.sqrt(simulation.beta/(2*Urms))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ,_
                    spec_k_rhines = simulation.spec[(np.abs(simulation.kr - k_rhines)).argmin()]
                    plt.loglog(np.array([k_rhines, k_rhines]), [spec_k_rhines/100, spec_k_rhines*100], colors[n][1])
                    plt.ylim([1e-8, 1e3])
                    simulation.drop_psi()
                    
            # plot inverse cascade slope
            
            cascade = 100*simulation.kr**(-5/3)
            plt.loglog(simulation.kr, cascade, 'k-.', label = '-5/3 slope')
            plt.legend()
            plt.title('Power spectra for different Res')
            plt.xlabel('kr')
            plt.ylabel('PSD')
            plt.savefig('1D_spectrum.png', dpi = 500, bbox_inches='tight', format = 'png')
            plt.show()
            
        def plt_dia(self, dia = 'energy'):
            ''' plots a diagnostic as a function of beta*L**3/nu (to observe phase change in turbulent behaviour)'''
            
            dias = np.zeros(len(self.iconvs))
            dias2 = np.zeros(len(self.iconvs))
            Res = np.zeros(len(self.iconvs)) #nondimensional betas
            dias_err = np.zeros(len(self.iconvs))
            dias_err2 = np.zeros(len(self.iconvs))
            
            if dia == 'orthogonality':
                dias = np.zeros([len(self.iconvs), 3])
            
            for n, Simulation in enumerate(self.simulations):
                Simulation.read_path(Iconv = self.iconvs[n])
                
                if dia == 'energy':
                    Simulation.calc_ke(plot = 0)
                    dias[n] = Simulation.ke_mean/(Simulation.L**2*Simulation.tau0**2/Simulation.beta**2) # adimensionalise
                    plt.ylabel('Total Energy')
                    
                if dia == 'energy_mean': #calculate energy of the mean flow
                    psi_mean = np.mean(Simulation.psi, axis = 0)
                    dias[n] = operators.energy(psi_mean[1:-1, 1:-1], operators.lap(psi_mean, Simulation.delta), Simulation.delta) 
                    dias[n] = dias[n]/(Simulation.tau0**2*Simulation.L**2/Simulation.beta**2) # adimensionalise
                    
                    psi_std = np.std(Simulation.psi, axis = 0)
                    var_der = -2*(psi_mean[2:,1:-1] + psi_mean[:-2,1:-1] + psi_mean[1:-1,2:] + psi_mean[1:-1,:-2] -4*psi_mean[1:-1, 1:-1]) #variational derivative of the mean energy is twice the vorticity
                    dias_err[n] = abs(np.sum(np.sum(var_der*psi_std[1:-1, 1:-1], axis = 1), axis = 0)) 
                    dias_err[n] = dias_err[n]/(Simulation.tau0**2*Simulation.L**2/Simulation.beta**2)
                    plt.ylabel('Energy of the mean flow')
                    
                if dia == 'dissipation':
                    Simulation.calc_diss(autocorr = 0, plot = 0)
                    dias[n] = Simulation.diss_mean/(Simulation.L*Simulation.tau0**2/Simulation.beta)
                    dias_err[n] = Simulation.diss_mean_err/(Simulation.L*Simulation.tau0**2/Simulation.beta)
                    plt.ylabel('total dissipation')
                    
                if dia == 'input':
                    Simulation.calc_input(autocorr = 0, plot = 0)
                    dias[n] = Simulation.input_mean/(Simulation.L*Simulation.tau0**2/Simulation.beta)
                    dias_err[n] = Simulation.input_mean_err/(Simulation.L*Simulation.tau0**2/Simulation.beta)
                    plt.ylabel('Energy input')
                    
                if dia == 'statvsdyna_diss':
                    Simulation.calc_diss_stat_vs_dyn(autocorr = 0)
                    dias[n] = Simulation.diss_stat/(Simulation.L*Simulation.tau0**2/Simulation.beta)
                    dias2[n] = Simulation.diss_fluc_mean/(Simulation.L*Simulation.tau0**2/Simulation.beta)
                    dias_err[n] = 0
                    dias_err2[n] = Simulation.diss_fluc_err/(Simulation.L*Simulation.tau0**2/Simulation.beta)
                    plt.ylabel('static/dynamic dissipation')
                    
                if dia == 'orthogonality': #tests for orthonality of mean flow with tau
                    simulation.test_orthogonality()
                    dias[n,:] = np.array([simulation.input, simulation.input_blue, simulation.input_red])/(simulation.L*simulation.tau0**2/simulation.beta)
                
                Res[n] = Simulation.Re
                Simulation.drop_psi()
                  
            if dia == 'energy_mean' or dia == 'enstrophy_mean':
                plt.plot(Res, dias)
                plt.errorbar(Res, dias, yerr = dias_err, ecolor = 'black', color = 'blue', fmt = "o")
                plt.xlabel(r'$\delta_M/\delta_I$')
                plt.show()
            elif dia == 'dissipation' or dia == 'input':
                #plt.figure(figsize = (8,6))
                plt.errorbar(Res, dias, yerr = dias_err, ecolor = 'black', color = 'blue', fmt = "o")
                plt.xlabel(r'$\delta_M/\delta_I$')
                plt.title('Finite dissipation experiment')
                #plt.savefig('fin_diss.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
            elif dia == 'statvsdyna_diss':
                #plt.figure(figsize = (8,6))
                plt.errorbar(Res, dias, yerr = dias_err, ecolor = 'black', color = 'blue', fmt = "o", label = 'mean flow dissipation')
                plt.errorbar(Res, dias2, yerr = dias_err2, ecolor = 'black', color = 'red', fmt = "o", label = 'dissipation due to dynamic flow features')
                plt.xlabel('Re')
                plt.ylabel('Dissipation')
                plt.legend()
                plt.title('Mean vs. Dynamic dissipation')
                #plt.savefig('mean_vs_dyn_diss.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
            elif dia == 'orthogonality':
                dias = np.array(dias)
                plt.plot(Res, dias[:,0], 'k', label = 'total input/dissipation')
                plt.plot(Res, dias[:,1], 'b', label = 'blue input')
                plt.plot(Res, dias[:,2], 'r', label = 'red input')
                plt.legend()
                plt.xlabel(r'$\delta_M')
                plt.ylabel('Energy input')
                plt.show() 
            else:
                plt.plot(Res, dias)
                plt.xlabel(r'$\delta_M')
                plt.show() 