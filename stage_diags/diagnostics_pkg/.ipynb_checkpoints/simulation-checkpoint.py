#diagnostics.py

from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import qgutils as qg
from matplotlib.animation import PillowWriter
import matplotlib.colors as mcolors
from diagnostics_pkg import operators
from diagnostics_pkg import utils
from scipy.optimize import curve_fit
from diagnostics_pkg import odysee
import copy
import gc
import xarray as xr
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.colors as colors
from diagnostics_pkg import build_modes_xarray

def find_nearest(a, a0):
    "Index of element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def log_normal(x, A, x0, sigma):
    return A * np.exp(-(np.log10(x) - x0) ** 2 / (2 * sigma ** 2))

class simulation:
        ''' Class to hold a simulation result and its parameters. '''
        def __init__(self, Path, Read_q = 0):
                self.path = Path
                self.N2 = 0
                self.read_q = Read_q
                
        def read_path(self, sizelim = 1e10, Iconv = 0):
                ''' function to read in data and variables for given simulation '''
        
                if len(self.path) != 2:
                    # single run experiments:
                        
                    # read in data
                    file2read = netcdf.NetCDFFile(f'{self.path}/outdir_0001/vars.nc','r','r')
                    self.psi = copy.deepcopy(file2read.variables['psi'][:,:,:])
                    
                    Shape = np.shape(self.psi)
                    Size = np.prod(Shape)
                    
                    if self.read_q == 1:
                        self.q = copy.deepcopy(file2read.variables['q'][:,:,:])
                        
                        Size = 2*Size
                        
                    self.time = copy.deepcopy(file2read.variables['time'][:])
                        
                    file2read.close()
                    del file2read
                    gc.collect()
                    
                    d = 1                   # cut down file size
                    if Size > sizelim:   
                            d = int(np.ceil(Size/sizelim))
                            if self.read_q == 1:
                                d *= 2
                                self.q = self.q[::d,:,:]
                            self.psi = self.psi[::d,:,:]    
                            Shape = np.shape(self.psi)
                    
                    
                    self.N = Shape[1] # grid size
                    # don't inverse for now
                    # self.Nt = Shape[0]# number of output files
                    # for n in range(self.Nt):
                    #         self.psi[n,:,:] = self.psi[n,:,:].T
                    
                    # if self.read_q == 1:
                    #     for n in range(self.Nt):
                    #             self.q[n,:,:] = self.q[n,:,:].T
                    
                    # read in parameters
                    ffile=open(f'{self.path}/params.in','r').read()
                        
                    if ffile[-7:] == 'vortex\n':
                        
                        self.tau0 = 0
                        
                            # Re
                        ini=ffile.find('Re')+(len('Re')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.Re = number
                        
                        
                            # Rh
                        ini=ffile.find('Rh')+(len('Rh')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.Rh = number
                            
                            # l0
                        ini=ffile.find('l0')+(len('l0')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.l0 = number
                        
                        
                            # psi0
                        ini=ffile.find('psi0')+(len('psi0')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.psi0 = number
                        
                            #L
                        ini=ffile.find('L0')+(len('L0')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.L = number
                        
                        self.nu = self.psi0/self.Re
                        self.beta = self.psi0/(self.Rh*self.l0**3)
                        
                    elif ffile[-8:] == 'highres\n':
                        print(2)
                        
                            # tau0
                        ini=ffile.find('tau0')+(len('tau0')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.tau0 = number
                        
                            # beta
                        ini=ffile.find('beta')+(len('beta')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.beta = number
                        
                            # nu
                        ini=ffile.find('nu')+(len('nu')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.nu = number
                        
                            # L0
                        ini=ffile.find('L0')+(len('L0')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.L = number
                        
                    
                    else: 
                            # tau0
                        ini=ffile.find('tau0')+(len('tau0')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.tau0 = number
                        
                            # beta
                        ini=ffile.find('beta')+(len('beta')+3)
                        rest=ffile[ini:]
                        search_enter=rest.find('\n')
                        number=float(rest[:search_enter])
                        self.beta = number
                        
                        #     # nu
                        # ini=ffile.find('nu')+(len('nu')+3)
                        # rest=ffile[ini:]
                        # search_enter=rest.find('\n')
                        # number=float(rest[:search_enter])
                        # self.nu = number
                        
                        #     # L0
                        # ini=ffile.find('L0')+(len('L0')+3)
                        # rest=ffile[ini:]
                        # search_enter=rest.find('\n')
                        # number=float(rest[:search_enter])
                        # self.L = number
                        
                            # eps_nl
                        ini = ffile.find('eps_nl')
                        if ini != -1:
                            ini +=(len('eps_nl')+3)
                            new = 0
                            rest=ffile[ini:]
                            search_enter=rest.find('\n')
                            number=float(rest[:search_enter])
                            self.eps_nl = number
                            
                            # eps_fr
                        ini=ffile.find('eps_fr = ')
                        if ini != -1:
                            ini += +(len('eps_fr')+3)
                            rest=ffile[ini:]
                            search_enter=rest.find('\n')
                            number=float(rest[:search_enter])
                            self.eps_fr = number
                            
                            # Re
                        ini=ffile.find('Re')
                        if ini != -1:
                            new = 1
                            ini += len('Re')+3
                            rest=ffile[ini:]
                            search_enter=rest.find('\n')
                            number=float(rest[:search_enter])
                            self.Re = number
                    
                            # delta_nl
                        ini=ffile.find('delta_nl')
                        if ini != -1:
                            ini += len('delta_nl')+3
                            rest=ffile[ini:]
                            search_enter=rest.find('\n')
                            number=float(rest[:search_enter])
                            self.delta_nl = number
                        
                            # iend (number of time steps in the simulation)
                        ini=ffile.find('iend')
                        if ini != -1:
                            ini += len('delta_nl')+3
                            rest=ffile[ini:]
                            search_enter=rest.find('\n')
                            number=int(float(rest[:search_enter]))
                            self.iend = number
                        
                        if new:
                            
                            self.L = (self.tau0/(self.delta_nl*self.beta)**2)**(1/3)
                            
                            self.nu = (self.Re**4/self.L**3/self.tau0)**(-1/2)
                        
                        else:
                            
                            self.L = (self.tau0/(self.eps_nl*self.beta)**2)**(1/3)
                            self.nu = self.eps_fr**3*self.beta*self.L**3
                            self.Re = (self.tau0*self.L**3/self.nu**2)**(1/4)
                            self.delta_nl = (self.tau0/(self.beta**2*self.L**3))**(1/2)
                        
                        # diout (time steps between outputs)
                    ini=ffile.find('iout')+(len('iout')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=int(rest[:search_enter])
                    
                    self.iout = number*d # include factor from downsampling due to memory limitations
                    
                    self.delta = self.L/(self.N - 1) # dx
                    
                    self.nconv = int(np.ceil(Iconv/self.iout))
                    
                    # cut out spin-off
                    
                    if self.read_q == 1:
                        q = self.q[self.nconv:,:,:]
                        del self.q
                        gc.collect()
                        self.q = q
                        
                    time = self.time[self.nconv:]
                    del self.time
                    gc.collect()
                    self.time = time
                    
                    psi = self.psi[self.nconv:,:,:]
                    del self.psi
                    gc.collect()
                    self.psi = psi
                    
                    self.Nt = len(self.psi[:,0,0])
                    
                    self.ispace = np.linspace(Iconv, Iconv + self.iout*self.Nt, self.Nt) # x axis for plots with step number
                    
                    self.nu_star = self.nu/np.sqrt(self.tau0*self.L**3)
                    self.beta_star = self.beta/np.sqrt(self.tau0/self.L**3)
                    
                    # self.iconv = Iconv
                    # self.nconv = int(np.ceil(self.iconv/self.iout))
                    # self.iend2 = 0
                    
                else: 
                    # multiple run experiments
                    
                    # read in data
                    file2read = netcdf.NetCDFFile(f'{self.path[0]}/outdir_0001/vars.nc','r','r')
                    self.psi = file2read.variables['psi'][:,:,:].copy() 
                    Shape = np.shape(self.psi)
                    Size = np.prod(Shape)
                    
                    if self.read_q == 1:
                        self.q = file2read.variables['q'][:,:,:].copy() 
                        Size = 2*Size
                        
                    self.time = file2read.variables['time'][:].copy()
                        
                    file2read.close()
                    
                    for n in range(1, len(self.path)):
                        file2read = netcdf.NetCDFFile(f'{self.path[n]}/outdir_0001/vars.nc','r','r')
                        psi_temp = file2read.variables['psi'][:,:,:].copy()
                        self.psi = np.concatenate(self.psi, psi_temp)
                        time_temp = file2read.variables['time'][:].copy()
                        self.time = np.concatenate(self.time, time_temp)
                        if self.read_q == 1:
                            q_temp = file2read.variables['q'][:,:,:].copy() 
                            self.q = np.concatenate(self.q, q_temp)
                        file2read.close()
                    
                    Shape = np.shape(self.psi)
                    Size = np.prod(Shape)
                    if self.read_q == 1:
                        Size = Size*2
                    
                    d = 1                   # cut down file size
                    if Size > sizelim:   
                            d = int(np.ceil(Size/sizelim))
                            self.psi = self.psi[::d,:,:]    
                            if self.read_q == 1:
                                self.q = self.q[::d,:,:]
                            Shape = np.shape(self.psi)
                    
                    self.N = Shape[1] # grid size
                    self.Nt = Shape[0]# number of output files
                    
                    for n in range(self.Nt):
                            self.psi[n,:,:] = self.psi[n,:,:].T
                    
                    if self.read_q == 1:
                        for n in range(self.Nt):
                                self.q[n,:,:] = self.q[n,:,:].T
                        
                        
                    # read in parameters
                    ffile=open(f'{self.path[0]}/params.in','r').read()
                    
                        # tau0
                    ini=ffile.find('tau0')+(len('tau0')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.tau0 = number
                    
                        # beta
                    ini=ffile.find('beta')+(len('beta')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.beta = number
                    
                        # eps_fr
                    ini=ffile.find('eps_fr = ')+(len('eps_fr')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.eps_fr = number
                    
                        # eps_nl
                    ini=ffile.find('eps_nl')+(len('eps_nl')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.eps_nl = number
                    
                    
                    self.L = (self.tau0/(self.eps_nl*self.beta)**2)**(1/3)
                    
                    self.nu = (self.eps_fr*self.L)**3*self.beta
                    
                    self.delta = self.L/self.N # dx
                    
                    ini=ffile.find('iout')+(len('iout')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=int(rest[:search_enter])
                    self.iout = number*d # include factor from downsampling due to memory limitations
                    
                    self.iconv = Iconv
                    self.nconv = int(np.ceil(self.iconv/self.iout))
                    
                    # cut out spin-off
                    
                    if self.read_q == 1:
                        self.q = self.q[self.nconv:,:,:]
                    
                    self.time = self.time[self.nconv:]
                   
                    self.psi = self.psi[self.nconv:,:,:]
                    
                    self.Nt = len(self.psi[:,0,0])
                    
                    self.ispace = np.linspace(Iconv, Iconv + self.iout*self.Nt, self.Nt) # x axis for plots with step number
                 
                gc.collect()
                
        def read_as_xarray(self, N_chunk, Iconv = 0):
            ''' reads the netcdffile as a chunked xarray object using dask. N_chunk is the number of matrix entries in one chunk '''
            
            # read in parameters
            ffile=open(f'{self.path}/params.in','r').read()
              
            if ffile[-8:] == 'highres\n':
                
                    # tau0
                ini=ffile.find('tau0')+(len('tau0')+3)
                rest=ffile[ini:]
                search_enter=rest.find('\n')
                number=float(rest[:search_enter])
                self.tau0 = number
                
                    # beta
                ini=ffile.find('beta')+(len('beta')+3)
                rest=ffile[ini:]
                search_enter=rest.find('\n')
                number=float(rest[:search_enter])
                self.beta = number
                
                    # nu
                ini=ffile.find('nu')+(len('nu')+3)
                rest=ffile[ini:]
                search_enter=rest.find('\n')
                number=float(rest[:search_enter])
                self.nu = number
                
                    # L0
                ini=ffile.find('L0')+(len('L0')+3)
                rest=ffile[ini:]
                search_enter=rest.find('\n')
                number=float(rest[:search_enter])
                self.L = number
            else:    
                    # tau0
                ini=ffile.find('tau0')+(len('tau0')+3)
                rest=ffile[ini:]
                search_enter=rest.find('\n')
                number=float(rest[:search_enter])
                self.tau0 = number
                
                    # beta
                ini=ffile.find('beta')+(len('beta')+3)
                rest=ffile[ini:]
                search_enter=rest.find('\n')
                number=float(rest[:search_enter])
                self.beta = number
                
                #     # nu
                # ini=ffile.find('nu')+(len('nu')+3)
                # rest=ffile[ini:]
                # search_enter=rest.find('\n')
                # number=float(rest[:search_enter])
                # self.nu = number
                
                #     # L0
                # ini=ffile.find('L0')+(len('L0')+3)
                # rest=ffile[ini:]
                # search_enter=rest.find('\n')
                # number=float(rest[:search_enter])
                # self.L = number
                
                    # eps_nl
                ini = ffile.find('eps_nl')
                if ini != -1:
                    ini +=(len('eps_nl')+3)
                    new = 0
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.eps_nl = number
                    
                    # eps_fr
                ini=ffile.find('eps_fr = ')
                if ini != -1:
                    ini += +(len('eps_fr')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.eps_fr = number
                    
                    # Re
                ini=ffile.find('Re')
                if ini != -1:
                    new = 1
                    ini += len('Re')+3
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.Re = number
            
                    # delta_nl
                ini=ffile.find('delta_nl')
                if ini != -1:
                    ini += len('delta_nl')+3
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=float(rest[:search_enter])
                    self.delta_nl = number
                
                    # iend (number of time steps in the simulation)
                ini=ffile.find('iend')
                if ini != -1:
                    ini += (len('iend')+3)
                    rest=ffile[ini:]
                    search_enter=rest.find('\n')
                    number=int(float(rest[:search_enter]))
                    self.iend = number
                    
                if new:
                    
                    self.L = (self.tau0/(self.delta_nl*self.beta)**2)**(1/3)
                    
                    self.nu = (self.Re**4/self.L**3/self.tau0)**(-1/2)
                
                else:
                    
                    self.L = (self.tau0/(self.eps_nl*self.beta)**2)**(1/3)
                    self.nu = self.eps_fr**3*self.beta*self.L**3
                    self.Re = (self.tau0*self.L**3/self.nu**2)**(1/4)
                    self.delta_nl = (self.tau0/(self.beta**2*self.L**3))**(1/2)
                
                # diout (time steps between outputs)
                ini=ffile.find('iout')+(len('iout')+3)
                rest=ffile[ini:]
                search_enter=rest.find('\n')
                number=int(rest[:search_enter])

            self.nu_star = self.nu/np.sqrt(self.tau0*self.L**3)
            self.beta_star = self.beta/np.sqrt(self.tau0/self.L**3)
            
            self.iout = number # include factor from downsampling due to memory limitations
          
            self.nconv = int(np.ceil(Iconv/self.iout))
            
            
            # create xarray, cut off the spinup and chunk it
            
            ds = xr.open_dataset(f'{self.path}/outdir_0001/vars.nc')
            
            self.Nt = len(ds['time']) - self.nconv
            self.N = len(ds['x'])
            self.N_chunk_time = int(N_chunk/(self.N**2))
            
            
            self.psi = ds['psi'][self.nconv:,:,:].chunk({'time': self.N_chunk_time})
            self.q = ds['q'][self.nconv:,:,:].chunk({'time': self.N_chunk_time})
            self.time = ds['time'][self.nconv:].chunk({'time': self.N_chunk_time})
            
            self.ispace = np.linspace(Iconv, Iconv + self.iout*self.Nt, self.Nt) # x axis for plots with step number
            self.delta = self.L/(self.N - 1) # dx
                
        ''' functions to extract snapshots and gifs '''
             
        def snap_q2(self, i):
                ''' calculates mean stream function and performs contour plot '''
                
                q2_snap = (self.q[i,:,:]**2)/2
                plt.figure(figsize = (8,8))
                plt.imshow(np.log10(q2_snap), origin = 'lower', vmin = 1, vmax = 3)
                plt.colorbar()
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
                plt.title(r'Enstrophy at $Re$ = ' + f'{(self.Re):.0f}' + r', $\delta_I$ = ' + f'{self.delta_nl:.2e}')
                # plt.savefig(f'mean_stream_Re_{(self.Re):.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
        def snap_q(self, i):
            if i == -1:
                n = -1
            else:
                n = find_nearest(self.ispace, i)
                
            q = operators.lap(self.psi[n,:,:], self.delta)
            plt.figure(figsize = (10,10))
            plt.imshow(q.T, origin = 'lower', cmap = 'seismic', vmin = -100, vmax = 100)
            plt.colorbar()
            plt.title('Snapshot of vorticity field at Re = 283')
            plt.xlabel('X (pixel)')
            plt.ylabel('Y (pixel)')
            plt.savefig('q_{self.eps_fr:.0e}.png', dpi = 500, bbox_inches='tight', format = 'png')
    
        def snap_psi(self, i):
            if i == -1:
                n = -1
            else:
                n = find_nearest(self.ispace, i)
            
            plt.imsave(f'psi_{self.eps_fr:.0e}.png', self.psi[n,:,:].T, origin = 'lower', format = 'png', dpi = 200, cmap = 'turbo')
            
        def gif_q(self, n_b = 0, n_t = 20):
            
            metadata = dict(title = 'gif', artist = 'Lenny')
            writer = PillowWriter(fps = 2, metadata = metadata)
            fig = plt.figure(figsize = (5,5))
            ax = fig.add_subplot(111, aspect = 1.0)
            plt.subplots_adjust(left = 0, bottom = 0.01, right = 0.99, top = 1)
            # qs = self.q[n_b:n_t+1,:,:].values
            maxx = 150
            # ax.set_title(r'Vorticity at $Re = $'+f'{self.Re**2:.1e}')
            ax.set_xticks([])
            ax.set_yticks([])
            
            with writer.saving(fig, f'q_{self.Re:.1e}.gif', 100):
                for n in range(0, n_t-n_b):
                    im = ax.imshow(self.q[n,(self.N- 3000):,:3000].values, origin = 'lower', cmap = 'seismic', vmin = -maxx, vmax = maxx)
                    # cb = fig.colorbar(im, ax = ax, fraction=0.046, pad=0.04, label = r'$\omega$')
                    writer.grab_frame()
                    # cb.remove()
                    
        def gif_psi(self, n_b = 0, n_t = 20):
            
            metadata = dict(title = 'gif', artist = 'Lenny')
            writer = PillowWriter(fps = 2, metadata = metadata)
            fig = plt.figure(figsize = (5,5))
            ax = fig.add_subplot(111, aspect = 1.0)
            plt.subplots_adjust(left = 0, bottom = 0.01, right = 0.99, top = 1)
            maxx = 20
            # ax.set_title(r'Stream function at $Re = $'+f'{self.Re**2:.1e}')
            ax.set_xticks([])
            ax.set_yticks([])
            
            with writer.saving(fig, f'psi_{self.Re:.1e}.gif', 100):
                for n in range(0, n_t-n_b):
                    im = ax.imshow(self.psi[n,:,:], origin = 'lower', cmap = 'seismic', vmin = -maxx, vmax = maxx)
                    #cb = fig.colorbar(im, ax = ax, fraction=0.046, pad=0.04, label = r'$\psi$')
                    writer.grab_frame()
                    # cb.remove()
        
        def movie_pres(self, Nt, filename):
            ''' makes movie to hang next to poster. Nt '''
            
            fig = plt.figure(figsize=(12, 6), dpi = 200)
            grid = plt.GridSpec(1, 2, hspace=0.3, wspace=0.3)
            psi_ax = fig.add_subplot(grid[0, 0])
            q_ax = fig.add_subplot(grid[0, 1])
            fig.suptitle(r'Simulation at $Re = 1.6 x 10^5$')
            
            psi_ax.set_xticks([])
            psi_ax.set_yticks([])
            
            q_ax.set_xticks([])
            q_ax.set_yticks([])
            
            v = 30 #np.max(abs(self.psi)).compute()
            v_q = 100 # np.max(abs(self.q)).compute()/20
            
            Cmap = 'seismic'

            # set up animation
            
            fps = 5    
            T = Nt/fps
            
            def make_frame(t):
                i = int(t*fps)
                
                # show it
                
                psi_plt = psi_ax.imshow(self.psi[i,:,:], vmin = -v, vmax = v, origin = 'lower', cmap = Cmap)
                q_plt = q_ax.imshow(self.q[i,:,:], vmin = -v_q, vmax = v_q, origin = 'lower', cmap = Cmap)
                cb1 = fig.colorbar(psi_plt, ax = psi_ax, fraction=0.046, pad=0.04, label = r'$\psi$')
                cb2 = fig.colorbar(q_plt, ax = q_ax, fraction=0.046, pad=0.04, label = r'$\omega$')
                
                ret = mplfig_to_npimage(fig)
                
                cb1.remove()
                cb2.remove()
                
                return ret

            # write animation and save it
            
            animation = VideoClip(make_frame, duration = T)
            animation.write_videofile(filename + '.mp4', fps = fps)
        
        def movie_en(self, Nt, filename,  i_start = 0):
            ''' makes movie of energy to hang next to poster. Nt '''
            
            fig = plt.figure(figsize=(6, 6), dpi = 200)
            grid = plt.GridSpec(1, 1, hspace=0.3, wspace=0.3)
            en_ax = fig.add_subplot(grid[0, 0])
            fig.suptitle(r'Simulation at $Re = 1.6 x 10^5$')
            
            en_ax.set_xticks([])
            en_ax.set_yticks([])
            
            v = 150 #np.max(abs(self.psi)).compute()
            vmin = 0
            
            Cmap = 'Blues'

            # set up animation
            
            fps = 2    
            T = Nt/fps
            levels = np.linspace(-30, 30, 10)
            
            def make_frame(t):
                i = int(t*fps)
                
                
                
                # show it
                en = np.zeros([self.N, self.N])
                en_in = ((self.psi[i_start + i,2:,1:-1].values - self.psi[i_start + i,:-2,1:-1].values)**2 + (self.psi[i_start + i,1:-1,2:].values - self.psi[i_start + i,1:-1,:-2].values)**2)/(2*self.delta)**2/2
                en[1:-1,1:-1] = en_in
                # norm=colors.LogNorm(vmin, vmax=v)
                en_plt = en_ax.imshow(en, vmax = v, vmin = 0, origin = 'lower', cmap = Cmap) # 
                # psi_plt = en_ax.contour(self.psi[i_start + i,:,:], origin = "lower", levels = levels, colors = 'k')
                cb1 = fig.colorbar(en_plt, ax = en_ax, fraction=0.046, pad=0.04, label = r'$\frac{u^2 + v^2}{2}$')
                
                ret = mplfig_to_npimage(fig)
                
                cb1.remove()
                plt.cla()
                
                return ret

            # write animation and save it
            
            animation = VideoClip(make_frame, duration = T)
            animation.write_videofile(filename + '.mp4', fps = fps)
        
        def movie_en_turb_threshold(self, Nt, filename,  i_start = 0):
            ''' makes movie of energy of turbulent flow features only. Apply Threshold-o-gram feature at every snapshot.'''
            
            # creat figure
            
            fig = plt.figure(figsize=(6, 6), dpi = 200)
            grid = plt.GridSpec(1, 1, hspace=0.3, wspace=0.3)
            en_ax = fig.add_subplot(grid[0, 0])
            fig.suptitle(rf'Simulation at $\nu^* = {self.nu/np.sqrt(self.tau0*self.L**3):.1e}$')
            
            en_ax.set_xticks([])
            en_ax.set_yticks([])
            
            Cmap = 'gnuplot'

            # set up animation
            
            fps = 2    
            T = Nt/fps
            psi_m = np.mean(self.psi, axis = 0).values
            q_m = np.mean(self.q, axis = 0).values
                        
            # wave numbers to be tested 
            waves = [[1,1], [1,2], [1,3],[2,1], [2,2], [2,3],[3,1]] 
                     
            
            def make_frame(t):
                i = int(t*fps)
                    
                # remove mean flow
                
                psi_dash = self.psi[i,:,:].values - psi_m
                q_dash = self.q[i,:,:].values - q_m
                psi_wave = np.zeros([self.N, self.N])
                
                # calculate wave field
                for wave in waves:
                    m = wave[0]
                    n = wave[1]
                    #project
                    psi_mn = build_modes_xarray.project_basin(q_dash, m, n, self.L)
                    phi = np.angle(psi_mn)
                    psi0 = np.abs(psi_mn)
                    # build mode
                    psi_wave_mn, psi_wave_BL, q_wave_mn, q_wave_BL = build_modes_xarray.build_basin(m, n, self.L, self.beta, self.nu, phi, self.N)
                    psi_wave += psi0*psi_wave_mn[0,:,:]
                
                # calculate turbulence field and energy
                psi_turb = psi_dash - psi_wave
                
                en = np.zeros([self.N, self.N])
                en_in = ((psi_turb[2:,1:-1] - psi_turb[:-2,1:-1])**2 + (psi_turb[1:-1,2:] - psi_turb[1:-1,:-2])**2)/(2*self.delta)**2/2
                en[1:-1,1:-1] = en_in
                
                # build threshol-o-gram
                
                Nbins = 250
                
                hist_tot = np.zeros(Nbins)
                bins = np.linspace(0,1000,Nbins + 1)
                
                en_interior = np.zeros((self.N-2)**2)
                en_bounds = np.zeros((self.N-2)*4)
                
                # transform fields into vectors and put it into histograms
                
                en_snap = en
                for n in range(1, self.N - 1):
                    en_bounds[4*(n-1)] = 0
                    en_bounds[4*(n-1) + 1] = 0
                    en_bounds[4*(n-1) +2] = 0
                    en_bounds[4*(n-1) +3] = 0

                    for m in range(1, self.N - 1):
                        en_interior[(n-1)*(self.N-2) + m-1] = en_snap[n,m]
                    
                        # create histograms
                    
                hist, bins = np.histogram(en_interior, bins)
                hist_bounds, bins = np.histogram(en_bounds, bins)
                
                hist_tot += np.array(hist) + hist_bounds/2
            
                
                integrant = (bins[:-1] + bins[1:])/2*hist_tot
                
                #calculate at which enstrophy p_crit of the dissipation happens
                
                integral = np.sum(integrant)
                
                cum = np.zeros(len(bins))
                
                cum[-1] = integrant[-1]
                
                i = -2
                
                for n in range(len(bins) - 2):
                    cum[i] = cum[i+1] + integrant[i]
                    i = i-1
                    
                cum /= integral
                
                cum[0] = 1
                
                threshold = np.zeros([self.N, self.N])
                
                for n in range(self.N):
                    for m in range(self.N):
                        idx = find_nearest(bins, en[n,m])
                        threshold[n,m] = cum[idx]
                        
                        # # plot regions in physical space
    
                        # plt.figure(figsize = (12,9))
                        # plt.imshow(threshold.T, cmap = 'nipy_spectral', vmin = 0, vmax = 1, origin = 'lower')
                        # plt.title(f'dissipation threshold-o-gram for $\delta_M = {(self.beta*self.L**3/self.nu)**-(1/3)/0.01:.2f} (snapshot)')
                        # plt.colorbar()
                        # plt.savefig(f'pic_dump_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}/diss_thresh_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}_{N}.png', dpi = 400, bbox_inches='tight', format = 'png')
                        # plt.show()
                    
                
                # show it
                
                threshold_plt = en_ax.imshow(threshold, vmax = 1, vmin = 0, origin = 'lower', cmap = Cmap) # 
                cb1 = fig.colorbar(threshold_plt, ax = en_ax, fraction=0.046, pad=0.04, label = r'$\frac{u^2 + v^2}{2}$')
                
                ret = mplfig_to_npimage(fig)
                
                cb1.remove()
                plt.cla()
                
                return ret

            # write animation and save it
            
            animation = VideoClip(make_frame, duration = T)
            animation.write_videofile(filename + '.mp4', fps = fps)
        
        ''' functions to extract scalar diagnostic value means and time series '''
        
        def calc_ke(self, autocorr = 1, plot = 1):
                '''function to calculate mean kinetic energy, determine its mean autocorrelation time and plot the data.'''
                
                # loop over array for memory reasons
                self.ke = np.zeros(self.Nt)
                for n in range(self.Nt):
                        if self.read_q == 1:
                            self.ke[n] = operators.energy(self.psi[n,:,:].values, self.q[n,:,:].values, self.delta)
                        else: 
                            self.ke[n] = operators.energy(self.psi[n,1:-1,1:-1].values, operators.lap(self.psi[n,:,:].values, self.delta), self.delta)
                
                if autocorr:
                      self.ke_mean, self.ke_mean_err = utils.corr_mean(self.ke) 
                else:
                        self.ke_mean = np.mean(self.ke)
                        self.ke_mean_err = np.std(self.ke)/np.sqrt(len(self.ke))
                
                print(f"Mean energy: {self.ke_mean} +- {self.ke_mean_err}")
                        
                if plot:
                    plt.plot(self.ispace, self.ke)
                    plt.xlabel('i')
                    plt.ylabel('Kinetic Energy')
                    plt.title(f'Kinetic Energy at Re = {self.Re}')
                    plt.show()
                
        def calc_diss(self, autocorr = 1, plot = 1):
                ''' function to plot (dimensional) dissipation as a function of time step '''
                
                # loop to save memory
                self.diss = np.zeros(self.Nt)
                for n in range(self.Nt):
                    self.diss[n] = -self.nu*2*operators.enstrophy(self.q[n,:,:].values, self.delta)
                    
                if autocorr:
                      self.diss_mean, self.diss_mean_err = utils.corr_mean(self.diss) 
                else:
                        self.diss_mean = np.mean(self.diss)
                        self.diss_mean_err = np.std(self.diss)/np.sqrt(len(self.diss))
                
                print(f"Mean dissipation: {self.diss_mean} +- {self.diss_mean_err}")
                        
                if plot:
                    plt.figure(figsize = (10,8))
                    plt.plot(self.ispace, self.diss, 'k')
                    plt.xlabel('t')
                    plt.ylabel('Dissipation')
                    plt.title(f'Dissipation at Re = {self.Re}')
                    plt.show()
                    
                    
        def calc_input(self, autocorr = 1, plot = 1):
                ''' function to calculate energy input through forcing.'''
        
                y_grid = np.zeros([self.N,self.N])
                for n in range(self.N):
                        y_grid[n,:] = np.linspace(0, self.L, self.N)
                Forcing = np.array(self.tau0*np.pi/self.L*np.sin(2*np.pi*y_grid/self.L)) # vorticity forcing (integrate normal forcing T*u by parts to get to this formalism)
                
                # loop to save memory
                self.input = np.zeros(self.Nt)
                for n in range(self.Nt):
                        self.input[n] = operators.integrate(self.psi[n,:,:].values*Forcing, self.delta)
                
                
                if autocorr:
                      self.input_mean, self.input_mean_err = utils.corr_mean(self.input) 
                else:
                        self.input_mean = np.mean(self.input)
                        self.input_mean_err = np.std(self.input)/np.sqrt(len(self.input))
                
                print(f"Mean energy input: {self.input_mean} +- {self.input_mean_err}")
                
                
                
                if plot:
                    #plt.hist(self.input/(self.L*self.tau0**2/self.beta), bins = 50)
                    #plt.show()
                    plt.plot(self.ispace, self.input)
                    plt.xlabel('i (step number)')
                    plt.ylabel('Input')
                    plt.title(r'Input at $Re$ = ' + f'{(self.Re):.2e}' + r', $\delta_I$ = ' + f'{self.delta_nl:.2e}')
                    plt.show()
        
        def calc_enstrophy_budget(self):
            ''' function that calculates a Reynolds-averaged budget of the enstrophy '''
            
            q_m = np.mean(self.q, axis = 0).values
            
            
            Z_turb_diss = []
            Z_turb_inflow = []

            for n in range(self.Nt):
                q = self.q[n,:,:].values - q_m
                Z_turb_diss.append(-self.nu*operators.palinstrophy(q, self.delta))
                Z_turb_inflow.append(self.nu*operators.enstrophy_bound(q, self.delta))
                
            Z_turb_diss = np.mean(Z_turb_diss)
            Z_turb_inflow = np.mean(Z_turb_inflow)
            
        
            y_grid = np.zeros([self.N,self.N])
            for n in range(self.N):
                    y_grid[n,:] = np.linspace(0, self.L, self.N)
            Forcing = np.array(-self.tau0*np.pi/self.L*np.sin(np.pi*y_grid/self.L)) # vorticity forcing (integrate normal forcing T*u by parts to get to this formalism)
            
            Z_forc = operators.integrate(q_m*Forcing, self.delta)
            
            Z_m_diss = -self.nu*operators.palinstrophy(q_m, self.delta)
            Z_m_inflow = self.nu*operators.enstrophy_bound(q_m, self.delta)
            
            return Z_turb_diss, Z_turb_inflow, Z_m_diss, Z_m_inflow, Z_forc
        
        def calc_diss_stat_vs_dyn(self, autocorr = 1):
                ''' function to plot (dimensional) dissipation of mean static flow as well as of the fluctuations as a function of time step '''
                
                q_m = np.mean(self.q, axis = 0)
                
                # loop to save memory
                self.diss_fluc = np.zeros(self.Nt)
                
                for n in range(self.Nt):
                        self.diss_fluc[n] = -self.nu*2*operators.enstrophy(self.q[n,:,:] - q_m, self.delta)
                
                self.diss_stat = -self.nu*2*operators.enstrophy(q_m, self.delta) # dissipation due to static flow
                
                if autocorr:
                      self.diss_fluc_mean, self.diss_fluc_err = utils.corr_mean(self.diss_fluc) 
                else:
                        self.diss_fluc_mean = np.mean(self.diss_fluc)
                        self.diss_fluc_err = np.std(self.diss_fluc)/np.sqrt(len(self.diss_fluc))
        
        ''' spectral analysis '''
        
        def calc_spec(self, window = 0):
                ''' calculates azimuthally averaged 1D energy spectrum '''
                
                dN = 1
                
                # Psi = self.psi
                # self.psi = Psi - np.mean(self.psi[self.nconv:,:,:], axis = 0)
                
                if window == 1:    
                    ''' plots power spectral density just over a small square in the turbulent region. Extend of the square is determined by px_b and px_t'''
                    
                    px_b = 0.0
                    px_t = 1
                    ix_b = int(self.N*px_b)
                    ix_t = int(self.N*px_t)
                    iy_b = int(self.N*(1-px_t))
                    iy_t = iy_b + ix_t - ix_b
                    
                    N_window = len(self.q[0,0,iy_b:iy_t])
                    N_spec = int(np.floor(N_window/2-1))
                    
                    self.spec = np.zeros(N_spec)
                    self.spec_hann = np.zeros(N_spec)
                    i = 0
                    
                    for n in range(0, self.Nt, dN):
                        
                        psi_interp = (self.psi[n, 1:, 1:] + self.psi[n, :-1, 1:] + self.psi[n, 1:, :-1] + self.psi[n, :-1, :-1])/4 
                        q_interp = (self.q[n, 1:, 1:] + self.q[n, :-1, 1:] + self.q[n, 1:, :-1] + self.q[n, :-1, :-1])/4 
                        
                        kr, spec_hann = qg.get_spec_1D(-psi_interp[ix_b:ix_t,iy_b:iy_t], q_interp[ix_b:ix_t,iy_b:iy_t], self.delta, window = "hanning")
                        kr, spec = qg.get_spec_1D(-psi_interp[ix_b:ix_t,iy_b:iy_t], q_interp[ix_b:ix_t,iy_b:iy_t], self.delta)
                        
                        self.spec = self.spec + spec
                        self.spec_hann = self.spec_hann + spec_hann
                        
                        i += 1
                        
                    self.kr = kr
                    self.spec = self.spec/i
                    self.spec_hann = self.spec_hann/i
                else:
                    
                    self.spec = np.zeros(int((self.N-3)/2))
                    i = 0
                    
                    for n in range(0, self.Nt, dN):
                            
                        psi_interp = (self.psi[n, 1:, 1:] + self.psi[n, :-1, 1:] + self.psi[n, 1:, :-1] + self.psi[n, :-1, :-1])/4 
                        q_interp = (self.q[n, 1:, 1:] + self.q[n, :-1, 1:] + self.q[n, 1:, :-1] + self.q[n, :-1, :-1])/4 
                        
                        kr, spec = qg.get_spec_1D(-psi_interp, q_interp, self.delta)
                        self.spec = self.spec + spec
                        i += 1
                        
                    self.spec = self.spec/i
                    self.kr = kr
                
                #k_Munk = 2*np.pi/(self.nu/self.beta)**(1/3)
                
                # plt.loglog(self.kr, self.spec, 'k-', label = 'PSD')
                # #plt.loglog(self.kr, self.spec_hann, 'b-', label = 'PSD (hanning windowed)')
                # plt.title('Power density spectrum as function of wavelength')
                # plt.xlabel('k')
                # plt.ylabel('PSD')
                
                # # calculate and plot Rhines scale
                # Urms = np.sqrt(np.sum(self.spec)*2*np.pi/self.L)
                # k_rhines = np.sqrt(self.beta/(2*Urms))
                # spec_k_rhines = self.spec[(np.abs(self.kr - k_rhines)).argmin()]
                # plt.loglog(np.array([k_rhines, k_rhines]), [spec_k_rhines/100, spec_k_rhines*100], '--g', label = 'Rhines scale')
                # print(k_rhines)
                
                
                # log_spec = np.log(self.spec)
                # log_kr = np.log(self.kr)
                # N_k = len(self.kr)
                # Boundfit = list(range(int(N_k/64), int(N_k/4)))
                # log_spec_fit = log_spec[Boundfit]
                # log_k_fit = log_kr[Boundfit]
                # z = np.polyfit(log_k_fit, log_spec_fit, 1)
                # print(f'spectral gradient was found to be {z[0]} at N = {self.N}')
                
                # k_fit = self.kr[Boundfit]
                # fit = np.exp(z[1])*k_fit**z[0]
                # plt.loglog(k_fit, fit, '-.y', label = 'fit')
                # plt.legend()
                # #plt.ylim([1e-10, 1e4])
                # plt.show()
                
        def calc_spec_2D(self, window = 0):
                ''' calculates 2D energy spectrum '''
                
                dN = 1
                
                if window == 1:    
                    ''' plots power spectral density just over a small square in the turbulent region. Extend of the square is determined by px_b and px_t'''
                    px_b = 0.1
                    px_t = 0.3
                    ix_b = int(self.N*px_b)
                    ix_t = int(self.N*px_t)
                    iy_b = int(self.N*(1-px_t))
                    iy_t = iy_b + ix_t - ix_b
                    
                    N_spec = len(self.q[0,ix_b:ix_t,iy_b:iy_t])
                    
                    self.spec = np.zeros([N_spec, N_spec])
                    i = 0
                    
                    for n in range(self.nconv + 1, self.Nt, dN):
                        
                        psi_interp = (self.psi[n, 1:, 1:] + self.psi[n, :-1, 1:] + self.psi[n, 1:, :-1] + self.psi[n, :-1, :-1])/4 
                        q_interp = (self.q[n, 1:, 1:] + self.q[n, :-1, 1:] + self.q[n, 1:, :-1] + self.q[n, :-1, :-1])/4 
                        
                        k, l, spec = qg.get_spec_2D(-psi_interp[ix_b:ix_t,iy_b:iy_t], q_interp[ix_b:ix_t,iy_b:iy_t], self.delta, window = "hanning")
                        self.spec = self.spec + spec
                        i += 1
                        
                    self.spec = self.spec/i
                    plt.imshow(np.log(np.abs(self.spec)))
                    plt.show()
                else:
                    
                    self.spec = np.zeros([int((self.N-1)), int((self.N-1))])
                    i = 0
                    
                    for n in range(0, self.Nt, dN):
                            
                        psi_interp = (self.psi[n, 1:, 1:] + self.psi[n, :-1, 1:] + self.psi[n, 1:, :-1] + self.psi[n, :-1, :-1])/4 
                        q_interp = (self.q[n, 1:, 1:] + self.q[n, :-1, 1:] + self.q[n, 1:, :-1] + self.q[n, :-1, :-1])/4 
                        
                        k, l, spec = qg.get_spec_2D(-psi_interp, q_interp, self.delta)
                        self.spec = self.spec + spec
                        i += 1
                        
                    self.spec = self.spec/i
                    # nn = 20 #only plot center region without log to maybe see the Rhine's bell
                    # plt.imshow((spec[int(self.N/2)-nn:int(self.N/2)+nn, int(self.N/2)-nn:int(self.N/2)+nn]))
                    
                    dk = k[0,1] - k[0,0]
                    N_ticks = 5
                    pos = np.linspace(0, self.N-1, N_ticks)
                    x = pos*dk - (self.N-1)/2*dk
                    
                    x = [f'{n:.1e}' for n in x]
                    
                    plt.imshow(np.log(np.abs(self.spec)), vmin = -30, vmax = 0, cmap = 'hot')
                    plt.xticks(pos, x)
                    plt.yticks(pos, x)
                    plt.xlabel('k')
                    plt.ylabel('l')
                    plt.colorbar()
                    plt.title(f'2D power spectrum for Re = {self.Re:.0f}')
                    plt.savefig(f'2D_spectrum_Re_{(self.Re):.0f}' +'.png', dpi = 500, bbox_inches='tight', format = 'png')
                    plt.show()
                    
                    center = int(self.N/2)
                    width = 15
                    N_ticks = 5
                    pos = np.linspace(0, 2*width - 1, N_ticks)
                    x = (pos - width + 0.5)*dk
                    
                    x = [f'{n:.1e}' for n in x]
                    
                    plt.imshow(np.log(np.abs(self.spec[center - width:center + width, center - width:center + width])), vmin = -2, vmax = 7, cmap = 'hot')
                    plt.xticks(pos, x)
                    plt.yticks(pos, x)
                    plt.xlabel('k')
                    plt.ylabel('l')
                    plt.colorbar()
                    plt.title(f'2D power spectrum for Re = {self.Re:.0f} (zoomed)')
                    plt.savefig(f'2D_spectrum_Re_zoomed_{(self.Re):.0f}' +'.png', dpi = 500, bbox_inches='tight', format = 'png')
                    plt.show()
        
        ''' 2D diagnostics '''
        
        def plt_psi_mean(self):
                ''' calculates mean stream function and performs contour plot '''
                
                psi_mean = np.mean(self.psi, axis = 0).values
                plt.figure(figsize = (5,5))
                plt.contour(psi_mean, colors = 'k', levels = np.linspace(-20, 20)*self.tau0/self.beta)
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
                # plt.title(r'Mean stream function at $Re$ = ' + f'{(self.Re):.0f}' + r', $\delta_I$ = ' + f'{self.delta_nl:.2e}')
                plt.savefig(f'mean_stream_Re_{(self.Re):.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
        def plt_Z_mean(self):
                ''' calculates mean stream function and performs contour plot '''
                
                en = np.zeros([self.N, self.N])
                for n in range(self.Nt):    
                    q = self.q[n,:,:].values
                    en += q**2
                en /= self.Nt
                
                plt.figure(figsize = (5,5))
                plt.imshow(en, origin = 'lower', cmap = 'Blues', vmin  = 0, vmax = 1000)
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
                plt.title(r'Mean Enstrophy at $Re$ = ' + f'{(self.Re):.0f}' + r', $\delta_I$ = ' + f'{self.delta_nl:.2e}')
                plt.savefig(f'mean_Z_Re_{(self.Re):.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.colorbar()
                plt.show()
           
                
        def plt_E_mean(self):
                ''' calculates mean stream function and performs contour plot '''
                
                en = np.zeros([self.N, self.N])
                for n in range(self.Nt):    
                    en_in = ((self.psi[n,2:,1:-1].values - self.psi[n,:-2,1:-1].values)**2 + (self.psi[n,1:-1,2:].values - self.psi[n,1:-1,:-2].values)**2)/(2*self.delta)**2/2
                    en[1:-1,1:-1] += en_in
                en /= self.Nt
                
                plt.figure(figsize = (5,5))
                plt.imshow(en, origin = 'lower', cmap = 'Blues')
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
                plt.title(r'Mean Energy at $Re$ = ' + f'{(self.Re):.0f}' + r', $\delta_I$ = ' + f'{self.delta_nl:.2e}')
                plt.savefig(f'mean_E_Re_{(self.Re):.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.colorbar()
                plt.show()
         
        def plt_mean_enst_flux(self):
                ''' plots enstrophy of the mean flow (omega_m**2/2) (flux as well as dissipation rates)''' # broken in new nodal code, not fixed yet
                
                for n in range(self.Nt):
                        self.psi[n,:,:] = self.psi[n,:,:].T
                
                #forcing
                y_grid = np.zeros([self.N,self.N])
                for n in range(self.N):
                        y_grid[n,:] = np.linspace(self.L/(2*self.N), self.L*(self.N-1)/(self.N), self.N)
                Forcing = -np.array(self.tau0*np.pi/self.L*np.sin(np.pi*y_grid/self.L)) # vorticity forcing (integrate normal forcing T*u by parts to get to this formalism)
                
                #calculate mean quantities
                omega_m = np.zeros([self.N,self.N])
                enst_m = np.zeros([self.N,self.N])
                u_m = np.zeros([self.N,self.N])
                v_m = np.zeros([self.N,self.N])
                omega_u_m = np.zeros([self.N,self.N])
                omega_v_m = np.zeros([self.N,self.N])
                
                for m in range(self.nconv, self.Nt):
                    omega = qg.laplacian(self.psi[m,1:-1, 1:-1], self.delta)
                    u, v = qg.comp_vel(self.psi[m,1:-1, 1:-1], self.delta)
                    omega_m += omega
                    enst_m += omega**2/2
                    u_m += u
                    v_m += v
                    omega_u_m += u*omega
                    omega_v_m += v*omega
                    
                omega_m /= -(self.nconv - self.Nt)
                enst_m /= -(self.nconv - self.Nt)
                u_m /= -(self.nconv - self.Nt)
                v_m /= -(self.nconv - self.Nt)
                omega_u_m /= -(self.nconv - self.Nt)
                omega_v_m /= -(self.nconv - self.Nt)
                
                # calculate gradient of omega (without knowing the bc for omega...)
                grad_omega_m = 1/(2*self.delta)*np.array([omega_m[1:-1,2:] - omega_m[1:-1,:-2], omega_m[2:,1:-1] - omega_m[:-2,1:-1]])
                place = np.zeros([2, self.N, self.N])
                place[:, 1:-1, 1:-1] = grad_omega_m
                grad_omega_m = place
                grad_omega_m[:,0,:] = grad_omega_m[:,1,:]
                grad_omega_m[:,-1,:] = grad_omega_m[:,-2,:]
                grad_omega_m[:,:,0] = grad_omega_m[:,:,1]
                grad_omega_m[:,:,-1] = grad_omega_m[:,:,-2]
                
                #calculate sources
                enst_forc = Forcing*omega_m
                m_diss = -self.nu*(grad_omega_m[0]**2 + grad_omega_m[1]**2)
                turb_exchange = (omega_u_m - omega_m*u_m)*grad_omega_m[0] + (omega_v_m - omega_m*v_m)*grad_omega_m[1]
                
                #calculate flux vectors
                omega_m_2_adv = np.array([u_m*omega_m**2/2, v_m*omega_m**2/2])
                beta_flux = self.beta*np.array([(v_m**2 - u_m**2)/2, -u_m*v_m])
                visc_flux = -self.nu*np.array([omega_m*grad_omega_m[0,:,:],omega_m*grad_omega_m[1,:,:]])  #-self.nu*grad_omega_m_2 
                turb_adv = np.array([(omega_u_m - omega_m*u_m)*omega_m, (omega_v_m - omega_m*v_m)*omega_m])
                
                #average spacially over sources and fluxes
                n_arr = 16
                n_av = int(self.N/n_arr)
                beta_flux_av = np.zeros([2, n_arr, n_arr])
                omega_m_2_adv_av = np.zeros([2, n_arr, n_arr])
                turb_adv_av = np.zeros([2, n_arr, n_arr])
                visc_flux_av = np.zeros([2, n_arr, n_arr])
                enst_forc_av = np.zeros([self.N, self.N])
                m_diss_av = np.zeros([self.N, self.N])
                turb_exchange_av = np.zeros([self.N, self.N])
                
                # average all flux vectors and sources over a small area
                for n in range(n_arr):
                    for m in range(n_arr):
                        beta_flux_av[0,n,m] = np.mean(beta_flux[0, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        beta_flux_av[1,n,m] = np.mean(beta_flux[1, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        omega_m_2_adv_av[0,n,m] = np.mean(omega_m_2_adv[0, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        omega_m_2_adv_av[1,n,m] = np.mean(omega_m_2_adv[1, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        turb_adv_av[0,n,m] = np.mean(turb_adv[0, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        turb_adv_av[1,n,m] = np.mean(turb_adv[1, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        visc_flux_av[0,n,m] = np.mean(visc_flux[0, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        visc_flux_av[1,n,m] = np.mean(visc_flux[1, n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        enst_forc_av[n,m] = np.mean(enst_forc[n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        m_diss_av[n,m] = np.mean(m_diss[n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        turb_exchange_av[n,m] = np.mean(turb_exchange[n*n_av:(n+1)*n_av, m*n_av:(m+1)*n_av])
                        
                # plot enstrophy lifecycle 
                sources = enst_forc_av + m_diss_av + turb_exchange_av
                sources_pos = np.zeros([n_arr, n_arr])
                sources_neg = np.zeros([n_arr, n_arr])
                for n in range(n_arr):
                    for m in range(n_arr):
                        if sources[n,m] > 0:
                            sources_pos[n,m] = sources[n,m]
                        else:
                            sources_neg[n,m] = -sources[n,m]
                            
                            
                flux = visc_flux_av + omega_m_2_adv_av + beta_flux_av + turb_adv_av
                flux_x, flux_y, max_len, min_len = transform(flux[0,:,:], flux[1,:,:])
                plt.quiver(flux_x, flux_y)
                plt.text(0,0,f'min = {min_len}, max = {max_len}')
                plt.imshow(np.log(sources_pos), origin = 'lower', cmap = 'Blues')
                plt.colorbar()
                plt.imshow(np.log(sources_neg), origin = 'lower', cmap = 'Reds')
                plt.colorbar()
                #plt.contour(psi_mean, colors = 'black')
                plt.title('Enstrophy lifecycle')
                plt.show()
                
                #plot individual sources
                plt.imshow(np.log(abs(m_diss + enst_forc + turb_exchange)), origin = 'lower')
                plt.title('All sources combined')
                plt.show()
                plt.imshow(np.log(abs(m_diss)), origin = 'lower')
                plt.title('viscous dissipation')
                plt.show()
                plt.imshow(np.log(abs(enst_forc)), origin = 'lower')
                plt.title('enstrophy forcing')
                plt.show()
                plt.imshow(np.log(abs(turb_exchange)), origin = 'lower')
                plt.title('turbulent enstrophy creation')
                plt.show()
                
                #plot individual fluxes
                plt.quiver(visc_flux_av[0,:,:], visc_flux_av[1,:,:])
                plt.title('viscous flux')
                plt.show()
                plt.quiver(beta_flux_av[0,:,:], beta_flux_av[1,:,:])
                plt.title('beta flux')
                plt.show()
                plt.quiver(omega_m_2_adv_av[0,:,:],omega_m_2_adv_av[1,:,:])
                plt.title('average advective flux')
                plt.show()
                plt.quiver(turb_adv_av[0,:,:],turb_adv_av[1,:,:])
                plt.title('turbulent advective flux')
                plt.show()
                
                for n in range(self.Nt):
                        self.psi[n,:,:] = self.psi[n,:,:].T
        
        def plt_diss_BL_scaling(self):
                ''' separates the global dissipation into dissipation happening in regions of the boundary layer thickness around the boundaries and the interior '''
                
                i_bl = int((self.nu/(self.beta*self.L**3))**(1/3)*self.N) # Munk Boundary Layer #int(5/self.Re*self.N)  # Prandtl Boundary Layer #int(2*np.sqrt(self.tau0/(self.beta**2*self.L**3))*self.N)#int(2*(self.nu/(self.beta*self.L**3))**(1/3)*self.N)
                
                print(i_bl)
                Nbins = 250
                
                hist_IN_tot = np.zeros(Nbins)
                hist_BL_tot = np.zeros(Nbins)
                logbins = np.logspace(-5,9,Nbins + 1)
                
                diss_interior = np.zeros((self.N-2*i_bl)**2)
                diss_bounds = np.zeros(2*(self.N-2)*(i_bl-1) + 2*(self.N - 2*(i_bl))*(i_bl-1))
                diss_bounds_b = np.zeros((self.N-2)*4)
                
                # transform fields into vectors and put it into histograms
                
                for N in range(self.Nt):
                    
                    # diss_snap = (self.q[N,:,:].values)**2
                    diss_snap = operators.calc_local_diss(self.psi[N,:,:].values, self.q[N,:,:].values, self.delta)
                    
                    i = 0 
                    j = 0  
                    
                    for n in range(1, self.N - 1):
                        diss_bounds_b[4*(n-1)] = diss_snap[0,n]
                        diss_bounds_b[4*(n-1) + 1] = diss_snap[-1,n]
                        diss_bounds_b[4*(n-1) +2] = diss_snap[n,0]
                        diss_bounds_b[4*(n-1) +3] = diss_snap[n,-1]
                                
                    for n in range(1, self.N-1):
                        for m in range(1, i_bl):
                            diss_bounds[i] = diss_snap[n,m]
                            i += 1
                            
                    for n in range(1, self.N - 1):
                        for m in range(self.N-i_bl, self.N-1):
                            diss_bounds[i] = diss_snap[n,m]
                            i += 1
                    
                    for m in range(i_bl, self.N-i_bl):
                        for n in range(1, i_bl):
                            diss_bounds[i] = diss_snap[n,m]
                            i += 1
                            
                    for m in range(i_bl, self.N-i_bl):
                        for n in range(self.N-i_bl, self.N-1):
                            diss_bounds[i] = diss_snap[n,m]
                            i += 1
                       
                    for m in range(i_bl, self.N - i_bl):
                        for n in range(i_bl, self.N - i_bl):
                            diss_interior[j] = diss_snap[n,m]
                            j += 1
    
                    hist_interior, bins = np.histogram(diss_interior, logbins)
                    hist_bounds, bins = np.histogram(diss_bounds, logbins)
                    hist_bounds_b, bins = np.histogram(diss_bounds_b, logbins)
                    
                    hist_IN_tot += hist_interior
                    hist_BL_tot += np.array(hist_bounds) + hist_bounds_b/2
                
                hist_BL_tot /= self.Nt # get the average histogram
                hist_IN_tot /= self.Nt
                
                print(np.sqrt(np.sum(hist_BL_tot) + np.sum(hist_IN_tot)))
                
                integrant_BL_tot = self.nu*(logbins[:-1] + logbins[1:])/2*hist_BL_tot*self.delta**2
                integrant_IN_tot = self.nu*(logbins[:-1] + logbins[1:])/2*hist_IN_tot*self.delta**2
                
                # lognormal fit    
                
                parameters, covariance = curve_fit(log_normal, (logbins[:-1] + logbins[1:])/2, hist_IN_tot, p0 = [1e5, 1e-5, 1-2])
                  
                fit_A = parameters[0]
                fit_x0 = parameters[1]
                fit_sigma = parameters[2]
                
                lognormal = log_normal(logbins, fit_A, fit_x0, fit_sigma)
                
                #plot histograms
                
                plt.plot(logbins, lognormal, color = 'red', label = 'Lognormal Fit')
                plt.stairs(hist_BL_tot, logbins, color = 'blue', alpha = 0.5, label = 'Boundary layer')
                plt.stairs(hist_IN_tot, logbins, color = 'red', alpha = 0.5, label = 'Interior structures')
                plt.stairs(hist_BL_tot + hist_IN_tot, logbins, color = 'black', alpha = 0.5, label = 'total dissipation')
                
                plt.title(r'Histogram of fluctuations (2*Munk-layer separation) for ' + f'Re = {self.Re:.0f}')
                # plt.xlim([1e-12, 1])
                # plt.ylim([1, 1e5])
                plt.xscale('log')
                #plt.yscale('log')
                plt.legend(loc = 'upper right')
                plt.savefig(f'diss_histo_BL_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
                # plot integrants for dissipation
                
                plt.stairs(integrant_BL_tot, logbins, color = 'blue', alpha = 0.5, label = 'Boundary layer')
                plt.stairs(integrant_IN_tot, logbins, color = 'red', alpha = 0.5, label = 'Interior structures')
                plt.stairs(integrant_IN_tot + integrant_BL_tot, logbins, color = 'black', alpha = 0.5, label = 'total dissipation')
                
                plt.title(r'dissipation through fluctuations (2*Munk-layer separation) for ' + f'Re = {self.Re:.0f}')
                # plt.xlim([1e-12, 1])
                # plt.ylim([0, 0.06])
                plt.xscale('log')
                plt.legend(loc = 'upper left')
                plt.savefig(f'diss_contribution_BL_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
                diss_IN = np.sum(integrant_IN_tot)
                diss_BL = np.sum(integrant_BL_tot)
                
                self.calc_diss(autocorr = 0, plot = 0)
                
                print('BL-scaling separation:')
                print(f'Average Dissipation by IN structures: {np.sum(diss_IN)}')
                print(f'Average Dissipation by BL structures: {np.sum(diss_BL)}')
                print(f'Total dissipation is {abs(self.diss_mean)}')
                print(f'Relative dissipation magnitude of BL wrt IN: {np.sum(integrant_BL_tot)/np.sum(integrant_IN_tot)}')   
                
                return diss_IN, diss_BL, hist_IN_tot, hist_BL_tot, logbins
            
        def plt_diss_int(self, plt_histo = 1, threshold_o_gram = 0, plt_Re = 0, plt_vortex_spec = 0):
                ''' determines the global threshold by integrating the PDF of q² in space and time of important dissipation events '''
                
                # create global and temporally integrated histograms to determine cutoff enstrophy
                
                Nbins = 250
                
                hist_tot = np.zeros(Nbins)
                logbins = np.logspace(-12,0,Nbins + 1)
                
                diss_interior = np.zeros((self.N-2)**2)
                diss_bounds = np.zeros((self.N-2)*4)
                
                # transform fields into vectors and put it into histograms
                
                for N in range(self.Nt):
                    diss_snap = self.nu*(self.q[N,:,:])**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                    for n in range(1, self.N - 1):
                        diss_bounds[4*(n-1)] = diss_snap[0,n]
                        diss_bounds[4*(n-1) + 1] = diss_snap[-1,n]
                        diss_bounds[4*(n-1) +2] = diss_snap[n,0]
                        diss_bounds[4*(n-1) +3] = diss_snap[n,-1]
    
                        for m in range(1, self.N - 1):
                            diss_interior[(n-1)*(self.N-2) + m-1] = diss_snap[n,m]
                        
                        # create histograms
                        
                    hist, bins = np.histogram(diss_interior, logbins)
                    hist_bounds, bins = np.histogram(diss_bounds, logbins)
                    
                    hist_tot += np.array(hist) + hist_bounds/2
                
                # lognormal fit    
                
                parameters, covariance = curve_fit(log_normal, (logbins[:-1] + logbins[1:])/2, hist_tot, p0 = [1e5, 1e-5, 1-2])
                  
                fit_A = parameters[0]
                fit_x0 = parameters[1]
                fit_sigma = parameters[2]
                
                lognormal = log_normal(logbins, fit_A, fit_x0, fit_sigma)
                
                # plot histogram and lognormal dist
                
                plt.plot(logbins, lognormal, color = 'red')
                plt.stairs(hist_tot, logbins, color = 'blue')
                plt.xscale('log')
                plt.title(r'Integrated histogram of fluctuations for $\delta_M$ = ' + f'{(self.beta*self.L**3/self.nu)**-(1/3):.2e}')
                plt.xlim([1e-12, 1])
                #plt.savefig(f'pdf_total_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
                # plot integrants for dissipation
                
                integrant = (logbins[:-1] + logbins[1:])/2*hist_tot
                
                plt.stairs(integrant, logbins, color = 'blue')
                plt.title(r'Time-integrated dissipation through fluctuations for $\delta_M$ = ' + f'{(self.beta*self.L**3/self.nu)**-(1/3):.2e}')
                plt.xlim([1e-12, 1])
                plt.xscale('log')
                #plt.savefig(f'diss_contribution_total_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
                #calculate at which enstrophy p_crit of the dissipation happens
                
                integral = np.sum(integrant)
                
                cum = np.zeros(len(logbins))
                
                cum[-1] = integrant[-1]
                
                i = -2
                
                for n in range(len(logbins) - 2):
                    cum[i] = cum[i+1] + integrant[i]
                    i = i-1
                    
                cum /= integral
                
                cum[0] = 1
                
                # loop to create threshold-o-grams
                
                if threshold_o_gram == 1:
                    for N in range(self.Nt):
                    
                        threshold = np.zeros([self.N, self.N])
                        
                        diss = self.nu*self.q[N,:,:]**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                        
                        for n in range(self.N):
                            for m in range(self.N):
                                idx = find_nearest(logbins, diss[n,m])
                                threshold[n,m] = cum[idx]
                                
                        # plot regions in physical space
    
                        plt.figure(figsize = (12,9))
                        plt.imshow(threshold.T, cmap = 'nipy_spectral', vmin = 0, vmax = 1, origin = 'lower')
                        plt.title(f'dissipation threshold-o-gram for $\delta_M = {(self.beta*self.L**3/self.nu)**-(1/3)/0.01:.2f} (snapshot)')
                        plt.colorbar()
                        plt.savefig(f'pic_dump_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}/diss_thresh_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}_{N}.png', dpi = 400, bbox_inches='tight', format = 'png')
                        plt.show()
                    
                # segment dissipative structures into BL and IN
                
                cutoff = 0.8
                
                diss_crit = logbins[find_nearest(cum, cutoff)]
                
                hist_BL_tot = np.zeros(len(logbins) - 1)
                hist_IN_tot = np.zeros(len(logbins) - 1)
                
                if plt_Re == 1:
                    Re_bins = np.linspace(-30000, 50000, 100)
                    Diss_bins = np.logspace(-5, 0, 100)
                    Re_hist = np.zeros(len(Re_bins) - 1)
                    Diss_hist = np.zeros(len(Diss_bins) - 1)
                    
                if plt_vortex_spec == 1:
                    Lower_m_size = 5 # minimal and maximal vortex size in pixels
                    Upper_m_size = 65
                    
                    N_ss = 6 # number of bins in which vortices are put
                    
                    Ls = [] # stores the radius of a vortex based on self similar arguments
                    Omega_ss = np.zeros([N_ss, Upper_m_size - Lower_m_size - 1]) # only store the points after the maximum in case the roundin doesn't work
                    Omega2_ss = np.zeros([N_ss, Upper_m_size - Lower_m_size - 1]) # only store the points after the maximum in case the roundin doesn't work
                    NN_ss = np.zeros([N_ss, Upper_m_size - Lower_m_size - 1])
                    d_ss = int((Upper_m_size - Lower_m_size)/N_ss)
                    
                    Omegas = []
                    Rhs = []
                    Res = []
                    
                    
                for N in range(self.Nt): 
                    # segmenation loop. First identifies critical dissipation areas, then separates them into BL and INterior
                    
                    diss_snap = self.nu*(self.q[N,:,:])**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                    
                    for i in range(self.N):
                        for j in range(self.N):
                            if diss_snap[i,j] > diss_crit:
                                diss_snap[i,j] = 1
                            else:
                                diss_snap[i,j] = 0
                    
                    # seperate into interior structures and boundary layers
                    
                    coor_list = odysee.search_islands(diss_snap)
                    
                    BL_coor_list = []
                    IN_coor_list = []
                    
                    for n in range(len(coor_list)):
                        coor_list[n] = np.array(coor_list[n])
                        if np.min(np.min(coor_list[n])) == 0 or np.max(np.max(coor_list[n])) == self.N-1:
                            BL_coor_list.append(coor_list[n])
                        else:
                            IN_coor_list.append(coor_list[n])
                    
                    BL = np.zeros([self.N, self.N])
                    IN = np.zeros([self.N, self.N])
                    
                    for BL_coors in BL_coor_list:
                        for coor in BL_coors:
                            BL[coor[0], coor[1]] = 1
                    
                    for IN_coors in IN_coor_list:
                        for coor in IN_coors:
                            IN[coor[0], coor[1]] = 1
                    
                    # test for circularity and store Re and dissipation
                    
                    if plt_Re == 1:
                        
                        Re = []
                        Diss = []
                        
                        diss_snap = self.nu*(self.q[N,:,:])**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                        
                        for IN_patch_coors in IN_coor_list:
                            circ = utils.circ_para(IN_patch_coors, 8)
                            if circ < 0.1:
                                diss = 0
                                re = 0
                                for coors in IN_patch_coors:
                                    diss += diss_snap[coors[0],coors[1]]
                                    re += self.q[N,coors[0], coors[1]]
                                # coors_list = np.array(IN_patch_coors)
                                # plt.scatter(coors_list[:,0], coors_list[:,1])
                                # plt.show()
                                re *= self.delta**2/self.nu
                                Re.append(re)
                                Diss.append(diss)
                                
                        re_hist, bins = np.histogram(Re, Re_bins)
                        diss_hist, bins = np.histogram(Diss, Diss_bins)
                        
                        Re_hist += re_hist
                        Diss_hist += diss_hist
                    
                    # test for circularity and store vortex parameters
                    
                    if plt_vortex_spec == 1:
                        
                        for IN_patch_coors in IN_coor_list:
                            
                            w_max = 0
                            for coors in IN_patch_coors:
                                if abs(self.q[N,coors[0],coors[1]]) > w_max:
                                    w_max = abs(self.q[N,coors[0],coors[1]])
                                    center = [coors[0], coors[1]]
                            rmax = 0
                            
                            for coors in IN_patch_coors:
                                r = ((coors[0] - center[0])**2 + (coors[1] - center[1])**2)**(1/2)
                                
                                if r > rmax:
                                    rmax = r
                                 
                            w_ll = np.zeros(int(rmax + 1)) # self similar radius
                            count_ll = np.zeros(int(rmax + 1))
                            A = len(w_ll)
                            
                            for coors in IN_patch_coors:
                                # self similar radius
                                r = ((coors[0] - center[0])**2 + (coors[1] - center[1])**2)**(1/2)
                                w_ll[int(np.floor(r))] += abs(self.q[N,coors[0], coors[1]])
                                count_ll[int(np.floor(r))] += 1
                                
                            
                            for n in range(A):
                                if count_ll[n] != 0:
                                    w_ll[n] /= count_ll[n]
                                    
                            ll = 0
                            for n in range(A):
                                if abs(w_ll[n]) > w_max/2:
                                    ll += 1
                                else:
                                    break
                            
                            if ll == A:
                                ll -=1
                            
                            if w_ll[ll] - w_max/2 < 0:
                                ll -= 1
                            
                            W_ll = w_ll[ll]
                                
                            # circularity check
                            
                            q2_av = 0
                            qF_av = 0
                            
                            i = 0
                            
                            for coors in IN_patch_coors:
                                r = int(np.floor(((coors[0] - center[0])**2 + (coors[1] - center[1])**2)**(1/2)))
                                if r < ll:
                                    qF_av += w_ll[r]*abs(self.q[N,coors[0],coors[1]])
                                    q2_av += self.q[N,coors[0],coors[1]]**2
                                    i += 1
                            
                            if i != 0:
                                qF_av /= i
                                q2_av /= i
                            
                                circ = np.sqrt(q2_av - qF_av)/np.sqrt(q2_av)
                            
                                if W_ll != 0 and ll < rmax-1 and circ < 0.05 and abs((W_ll-w_max/2)*2/w_max) < 0.1:
                                    
                                    plt.plot(w_ll[:ll]/w_max)
                                    plt.plot(w_ll[:ll]/w_ll[:ll]/2)
                                    plt.title(f'{circ}')
                                    plt.show()
                                    
                                    min_x = np.min(IN_patch_coors[:,0])
                                    max_x = np.max(IN_patch_coors[:,0])
                                    min_y = np.min(IN_patch_coors[:,1])
                                    max_y = np.max(IN_patch_coors[:,1])
                                    plt.imshow(self.q[N,min_x:max_x, min_y:max_y].T, origin = 'lower')
                                    plt.title(f'{circ}')
                                    plt.show()
                                    
                                    # vortex profiling. For every vortex we first rescale, then count in order not to double count resized values that fall into the same bin due to rounding errors.
                                    
                                    n_ss = int(np.floor((ll-Lower_m_size)/d_ss))
                                    if n_ss >= 0 and n_ss < N_ss:
                                        Omega_ss_temp = np.zeros(Upper_m_size - Lower_m_size - 1)
                                        NN_ss_temp = np.zeros(Upper_m_size - Lower_m_size - 1)
                                        for m_ss in range(0,ll):
                                            mm_ss = int(np.floor(m_ss/ll*(n_ss*d_ss + Lower_m_size)))
                                            if mm_ss > 0 and mm_ss < ll:
                                                mm_ss -= 1
                                                Omega_ss_temp[mm_ss] += w_ll[m_ss]/w_max
                                                NN_ss_temp[mm_ss] += 1
                                        
                                        for n in range(len(Omega_ss_temp)):
                                            if NN_ss_temp[n] != 0:
                                                Omega_ss_temp[n] = Omega_ss_temp[n]/NN_ss_temp[n]
                                        
                                        Omega_ss[n_ss,:] += Omega_ss_temp
                                        Omega2_ss[n_ss,:] += Omega_ss_temp**2
                                        for m in range(n_ss*d_ss + Lower_m_size - 1):
                                            NN_ss[n_ss,m] += 1
                                        
                                        # dimensionalising and calculation of Rh and Re
                                        
                                        ll *= self.delta
                                        
                                        ll_nd = ll/self.L
                                        omega_nd = w_max/(self.tau0/(self.beta*self.L**2))
                                        
                                        Ls.append(ll_nd)
                                        Omegas.append(abs(omega_nd))
                                        Rhs.append(w_max/(self.beta*ll))
                                        Res.append(w_max*ll**2/self.nu)
                                        
                                    
                    # create boundary and interior dissipation histograms
                    
                    if plt_histo == 1:
                        
                        diss_snap = self.nu*(self.q[N,:,:])**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                        
                        diss_IN = []
                        
                        for IN_patch_coors in IN_coor_list:
                            diss_patch = list(np.zeros(len(IN_patch_coors)))
                            for n, coors in enumerate(IN_patch_coors):
                                diss_patch[n] = diss_snap[coors[0], coors[1]]
                            diss_IN = diss_IN + diss_patch
                            
                        diss_BL = []
                        diss_BL_boundary = []
                        
                        for BL_patch_coors in BL_coor_list:
                            diss_patch = list(np.zeros(len(BL_patch_coors)))
                            diss_patch_boundary = list(np.zeros(self.N*4))
                            i = 0
                            j = 0
                            for coors in BL_patch_coors:
                                if coors[0] == 0 or coors[0] == self.N-1 or coors[1] == 0 or coors[1] == self.N-1: # check if point is on the boundary
                                    diss_patch_boundary[i] = diss_snap[coors[0], coors[1]]
                                    i += 1
                                else:
                                    diss_patch[j] = diss_snap[coors[0], coors[1]]
                                    j += 1
                            diss_patch = diss_patch[:j]
                            diss_patch_boundary = diss_patch_boundary[:i]
                            
                            diss_BL = diss_BL + diss_patch
                            diss_BL_boundary = diss_BL_boundary + diss_patch_boundary
                            
                        hist_IN, bins = np.histogram(diss_IN, logbins)
                        
                        hist_BL, bins = np.histogram(diss_BL, logbins)
                        hist_BL_boundary, bins = np.histogram(diss_BL_boundary, logbins)
                        
                        hist_BL = np.array(hist_BL) + np.array(hist_BL_boundary)/2
                            
                        hist_BL_tot += hist_BL
                        hist_IN_tot += hist_IN
                
                if plt_Re == 1:
                    
                    plt.stairs(Re_hist, Re_bins)
                    plt.xlabel('local vortex Re')
                    plt.ylabel('Number of occurences')
                    plt.title('local Re histogram')
                    plt.show()
                    
                    plt.stairs(Diss_hist, Diss_bins)
                    plt.xscale('log')
                    plt.xlabel('Dissipation associated to vortices')
                    plt.ylabel('Number of occurences')
                    plt.title('Dissipation integrated over all vortices in the flow')
                    plt.show()
                    
                if plt_vortex_spec == 1:
                    
                    for m in range(Upper_m_size - Lower_m_size -1):
                        for n in range(N_ss):
                            if NN_ss[n,m] != 0:
                                Omega_ss[n,m] /= NN_ss[n,m]
                                Omega2_ss[n,m] /= NN_ss[n,m]
                    
                    Err_Omega_ss = np.sqrt(Omega2_ss - Omega_ss**2)
                    
                    plt.figure(figsize = (0.9*10,1.4*10))
                    for n in range(N_ss):
                        l_max = n*d_ss + Lower_m_size - 1
                        start = 1/l_max
                        plt.errorbar(np.linspace(start,1,l_max),Omega_ss[n,:l_max], yerr = Err_Omega_ss[n, :l_max], label = f'r_min = {(n*d_ss + Lower_m_size)*self.delta/self.L:.1e} r_max = {((n+1)*d_ss + Lower_m_size)*self.delta/self.L:.1e} ')
                    plt.xlim([-0.2,1.2])
                    plt.ylim([0.3, 1.2])
                    plt.legend()
                    plt.ylabel('vorticity (normalised by w_max)')
                    plt.xlabel('r (rescaled with determined vortex width)')
                    plt.title(f'Self-similar vortex profiles at Re = {self.Re:.0f}')
                    plt.gca().set_aspect((0.9)/(1.4))
                    plt.savefig(f'vortex_profiles_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                    
                    plt.show()
                    
                    w_crit_dim = np.sqrt(diss_crit*(self.tau0**2/self.beta*self.L)/self.delta**2/self.nu)/(self.tau0/(self.beta*self.L**2)) # first transform to dimensional q2_max, then take sqrt, then transform to non-dimensional q_max with respect to Sverdrup
                    
                    plt.plot(np.linspace(5e-3,5e-2, 50), np.ones(50)*w_crit_dim, 'k--', label = 'minimim vorticity from segmentation algorithm')
                    plt.plot(np.linspace(5e-3,5e-2, 50), 5000*np.linspace(5e-3,5e-2, 50)**(-2/3), 'r-.', label = 'Kolmogorov scaling')
                    plt.scatter(Ls, Omegas, label = 'vortices self similar radius', marker = "1")
                    plt.legend()
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlim([5e-3,5e-2])
                    plt.ylim([5e3, 5e5])
                    plt.ylabel('Max vorticity (vs Sverdrup)')
                    plt.xlabel('characteristic length (vs size of domain)')
                    plt.title(f'Vortex spectrum at Re = {self.Re:.0f}')
                    
                    # log_spec = np.log(Omegas)
                    # log_kr = np.log(Ls)
                    # z = np.polyfit(log_kr, log_spec, 1)
                    # plt.text(np.mean(Ls), np.mean(Omegas), f'fitted exponent: {z[0]:.2f}')
                    plt.savefig(f'vortex_spectrum_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                    plt.show()
                    
                    # add cutoff due to numerical sampling restrictions
                    vortex_munk_sample = Lower_m_size**3*self.delta**3*self.beta/self.nu
                    
                    x_range = np.linspace(1e2, 1e5)
                    plt.plot(x_range, x_range/vortex_munk_sample, 'b--', label = 'Lower vortex sampling limit')
                    plt.plot(x_range, x_range, 'k--', label = 'Vortex Munk = 1')
                    plt.scatter(Res, Rhs, label = 'vortices', marker = "1")
                    plt.legend()
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlim([1e2,1e5])
                    plt.ylim([1e1, 1e4])
                    plt.ylabel('vortex Rh')
                    plt.xlabel('vortex Re')
                    plt.title(f'Vortex phase space at Re = {self.Re:.0f}')
                    plt.savefig(f'vortex_phase_space_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                    
                    plt.show()
                    
                if plt_histo == 1:
                        
                    hist_BL_tot /= self.Nt # get the average histogram
                    hist_IN_tot /= self.Nt
                    
                    integrant_BL_tot = (logbins[:-1] + logbins[1:])/2*hist_BL_tot
                    integrant_IN_tot = (logbins[:-1] + logbins[1:])/2*hist_IN_tot
                    
                    #plot histograms
                    
                    plt.stairs(hist_BL_tot, logbins, color = 'blue', alpha = 0.5, label = 'Boundary layer')
                    plt.stairs(hist_IN_tot, logbins, color = 'red', alpha = 0.5, label = 'Interior structures')
                    plt.stairs(hist_BL_tot + hist_IN_tot, logbins, color = 'black', alpha = 0.5, label = 'total dissipation')
                    
                    plt.title(r'Histogram of fluctuations (threshold islands separation) for $\delta_M$ = ' + f'{(self.beta*self.L**3/self.nu)**-(1/3):.2e}')
                    plt.xlim([1e-12, 1])
                    plt.ylim([1, 1e5])
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.legend(loc = 'upper right')
                    plt.savefig(f'diss_histo_island_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}.png', dpi = 500, bbox_inches='tight', format = 'png')
                    plt.show()
                    # plot integrants for dissipation
                    
                    plt.stairs(integrant_BL_tot, logbins, color = 'blue', alpha = 0.5, label = 'Boundary layer')
                    plt.stairs(integrant_IN_tot, logbins, color = 'red', alpha = 0.5, label = 'Interior structures')
                    plt.stairs(integrant_IN_tot + integrant_BL_tot, logbins, color = 'black', alpha = 0.5, label = 'total dissipation')
                    
                    plt.title(r'Dissipation of fluctuations (threshold islands separation) for $\delta_M$ = ' + f'{(self.beta*self.L**3/self.nu)**-(1/3):.2e}')
                    plt.xlim([1e-10, 1e-2])
                    plt.ylim([0, 0.06])
                    plt.xscale('log')
                    plt.legend(loc = 'upper left')
                    plt.savefig(f'diss_contribution_island_{(self.beta*self.L**3/self.nu)**-(1/3):.2e}.png', dpi = 500, bbox_inches='tight', format = 'png')
                    plt.show()
                    
                    print('Island separation:')
                    print(f'Average Dissipation by IN structures: {np.sum(integrant_IN_tot)}')
                    print(f'Average Dissipation by BL structures: {np.sum(integrant_BL_tot)}')
                    print(f'Relative dissipation magnitude of BL wrt IN: {np.sum(integrant_BL_tot)/np.sum(integrant_IN_tot)}')     
        
        def plt_enst_bound(self, diss_low, diss_high):
                ''' plots binary images of q with whites areas lying withing the range given to the function (in non-dimensionalised values by sverdrup)'''
                
                for n in range(self.Nt):
                    binary = np.zeros([self.N, self.N])
                    diss_snap = self.nu*(self.q[n,:,:])**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                    for n in range(self.N):
                        for m in range(self.N):
                            if diss_snap[n,m] < diss_high and diss_snap[n,m] > diss_low:
                                binary[n,m] = 1
                    
                    plt.figure(figsize = (10, 10))
                    plt.imshow(binary.T, origin = 'lower', cmap = 'Greys')
                    plt.show()
                    
        def enst_area_frac(self, q2_low):
                ''' calculates fractional area of q^2 values lying above a given value q2_low. '''
                
                Area_frac = 0
                
                for i in range(self.Nt):
                    binary = np.zeros([self.N, self.N])
                    q2_snap = np.array(self.q[i,:,:].values**2)
                    
                    binary = q2_snap > q2_low
                    binary_sum = binary.sum()
                    
                    # for n in range(self.N):
                    #     for m in range(self.N):
                    #         if q2_snap[n,m] > q2_low:
                    #             binary[n,m] = 1
                    if i == 1:
                        binary_out = binary
                        
                    Area_frac += binary_sum
                    
                Area_frac /= (self.Nt*self.N**2)
                    
                return Area_frac, binary_out
        
        def plt_diss_av_snap(self):
                ''' plots local dissipation (nu*integral of omega^2), in adimensional (normalised by sverdrup input)'''
                        
                q2_m = np.zeros([self.N,self.N])
                
                for m in range(self.Nt):
                    q2_m += self.q[m,:,:]**2
                    
                q2_m /= self.Nt
                omega_m = np.mean(self.q, axis = 0)
                
                diss_mean  = self.nu*q2_m/(self.tau0**2/self.beta*self.L)*self.delta**2
                diss_mean_flow = self.nu*omega_m**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                diss_fluc = (diss_mean - diss_mean_flow)
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 5))
                fig.suptitle(r'Energy Dissipation for Re = {self.Re:.0f}')
                
                norm = mcolors.TwoSlopeNorm(vmin=-2, vmax = 2, vcenter=0)
                
                pl1 = ax1.imshow(np.log10(diss_mean.T), cmap = 'nipy_spectral', origin = 'lower', norm = norm)
                ax1.axis('off')
                plt.colorbar(pl1,ax=ax1)
                ax1.title.set_text('log(Mean Dissipation)')
                
                pl2 = ax2.imshow(np.log10(diss_mean_flow.T), cmap = 'nipy_spectral', origin = 'lower', norm = norm)
                ax2.title.set_text('log(Dissipation due to Mean flow)')
                plt.colorbar(pl2,ax=ax2)
                ax2.axis('off')
                
                pl3 = ax3.imshow(np.log10(diss_fluc.T), cmap = 'nipy_spectral', origin = 'lower', norm = norm)
                ax3.title.set_text('log(Dissipation due to fluctuations)')
                plt.colorbar(pl3,ax=ax3)
                ax3.axis('off')
                
                #plt.savefig(f'diss_loc_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                
                plt.show()
                
                diss_mean_interior = np.zeros((self.N-2)**2)
                diss_snap_interior = np.zeros((self.N-2)**2)
                diss_mean_bounds = np.zeros((self.N-2)*4) # boundaries need to be stored seperately because they only occupy half a stencil
                diss_snap_bounds = np.zeros((self.N-2)*4) 
                
                diss_snap= self.nu*self.q[-1,:,:]**2/(self.tau0**2/self.beta*self.L)*self.delta**2
                
                # transform fields into vectors
                for n in range(1, self.N - 1):
                    diss_mean_bounds[4*(n-1)] = diss_mean[0,n]
                    diss_mean_bounds[4*(n-1) + 1] = diss_mean[-1,n]
                    diss_mean_bounds[4*(n-1) +2] = diss_mean[n,0]
                    diss_mean_bounds[4*(n-1) +3] = diss_mean[n,-1]

                    diss_snap_bounds[4*(n-1)] = diss_snap[0,n]
                    diss_snap_bounds[4*(n-1) + 1] = diss_snap[-1,n]
                    diss_snap_bounds[4*(n-1) +2] = diss_snap[n,0]
                    diss_snap_bounds[4*(n-1) +3] = diss_snap[n,-1]

                    for m in range(1, self.N - 1):
                        diss_snap_interior[(n-1)*(self.N-2) + m-1] = diss_snap[n,m]
                        diss_mean_interior[(n-1)*(self.N-2) + m-1] = diss_mean[n,m]
                        
                logbins = np.logspace(-10,0,1000)
                
                # create histograms
                
                mean_hist, bins = np.histogram(diss_mean_interior, logbins)
                snap_hist, bins = np.histogram(diss_snap_interior, logbins)
                
                mean_hist_bounds, bins = np.histogram(diss_mean_bounds, logbins)
                snap_hist_bounds, bins = np.histogram(diss_snap_bounds, logbins)
                
                mean_hist = np.array(mean_hist) + mean_hist_bounds/2
                snap_hist = np.array(snap_hist) + snap_hist_bounds/2
                
                # plot histograms
                
                plt.stairs(mean_hist, logbins, color = 'blue', alpha = 0.5, label = 'time average')
                plt.stairs(snap_hist, logbins, color = 'red', alpha = 0.5, label = 'snapshot')
                
                plt.xscale('log')
                
                plt.title(f'Dissipation PDF for Re = {self.Re:.0f}')
                plt.xlim([1e-10, 1e-0])
                plt.legend()
                #plt.savefig(f'pdf_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
                integrant_mean = (logbins[:-1] + logbins[1:])/2*mean_hist
                integrant_snap = (logbins[:-1] + logbins[1:])/2*snap_hist
                
                # plot integrants for dissipation
                
                plt.stairs(integrant_mean, logbins, color = 'blue', alpha = 0.5, label = 'time average')
                plt.stairs(integrant_snap, logbins, color = 'red', alpha = 0.5, label = 'snapshot')
                
                plt.title(f'Dissipation contribution for Re = {self.Re:.0f}')
                plt.xlim([1e-10, 1e-2])
                plt.xscale('log')
                plt.legend()
                #plt.savefig(f'diss_contribution_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
                # calculate at which enstrophy p_crit of the dissipation happens
                
                int_mean = np.sum(integrant_mean)
                int_snap = np.sum(integrant_snap)
                
                int_snap_dir = self.nu*2*operators.enstrophy(self.q[-1,:,:], self.delta)/(self.tau0**2/self.beta*self.L)
                
                # check that dissipation integral yields the same value
                print(int_snap)
                print(int_snap_dir)
                
                cum_snap = np.zeros(len(logbins))
                cum_mean = np.zeros(len(logbins))
                
                cum_snap[-1] = integrant_snap[-1]
                cum_mean[-1] = integrant_mean[-1]
                
                i = -2
                
                for n in range(len(logbins) - 2):
                    cum_snap[i] = cum_snap[i+1] + integrant_snap[i]
                    cum_mean[i] = cum_mean[i+1] + integrant_mean[i]
                    i = i-1
                    
                cum_mean /= int_mean
                cum_snap /= int_snap
                
                
                cum_snap[0] = 1
                cum_mean[0] = 1
                
                threshold_snap = np.zeros([self.N, self.N])
                threshold_mean = np.zeros([self.N, self.N])
                
                for n in range(self.N):
                    for m in range(self.N):
                        idx = find_nearest(logbins, diss_snap[n,m])
                        threshold_snap[n,m] = cum_snap[idx]
                        
                        idx = find_nearest(logbins, diss_mean[n,m])
                        threshold_mean[n,m] = cum_mean[idx]
                        
                # plot regions in physical space

                plt.figure(figsize = (12,9))
                plt.imshow(threshold_snap.T, cmap = 'nipy_spectral', origin = 'lower')
                plt.title(f'dissipation threshold-o-gram for $\delta_M = {(self.beta*self.L**3/self.nu)**-(1/3)/0.01:.2f} (snapshot)')
                plt.colorbar()
                #plt.savefig(f'diss_threshold_snap_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
                plt.figure(figsize = (12,9))
                plt.imshow(threshold_mean.T, cmap = 'nipy_spectral', origin = 'lower')
                plt.title(f'dissipation threshold-o-gram for Re = {self.Re:.0f} (time average)')
                plt.colorbar()
                plt.savefig(f'diss_threshold_mean_Re_{self.Re:.0f}.png', dpi = 500, bbox_inches='tight', format = 'png')
                plt.show()
                
        def test_orthogonality(self):
                ''' function that plots the mean stream function in a diverging colorbar 
                and then looks at the integrals of vorticity and energy forcing of the positive (red)
                and the negative (blue) parts separately.'''
                
                psi_mean = np.mean(self.psi, axis = 0)
                norm = mcolors.TwoSlopeNorm(vmin=-4, vmax = 4, vcenter=0)
                plt.imshow(psi_mean.T, cmap = 'seismic', norm = norm, origin = 'lower')
                plt.colorbar()
                plt.title(r'Mean stream function at $\delta_M/\delta_I$ = ' + f'{(self.beta*self.L**3/self.nu)**-(1/3)/0.01:.2f}')
                plt.show()
                
                vortex = [] # vortex in the northwestern corner
                rest = []
                
                for n in range(self.N):
                    for m in range(self.N):
                        if psi_mean[n,m] < 0:
                            vortex.append([n,m])
                        else:
                            rest.append([n,m])
                
                y_grid = np.zeros([self.N,self.N])
                for n in range(self.N):
                        y_grid[n,:] = np.linspace(self.L/(2*self.N), self.L*(self.N-1)/(self.N), self.N)
                Forcing = np.array(self.tau0*np.pi/self.L*np.sin(np.pi*y_grid/self.L)) # vorticity forcing (integrate normal forcing T*u by parts to get to this formalism)
                
                Forcing_blue = 0
                Forcing_red = 0
                
                for n in range(len(vortex)):
                    Forcing_blue += Forcing[vortex[n][0], vortex[n][1]]*psi_mean[vortex[n][0], vortex[n][1]]
                    
                for n in range(len(rest)):
                    Forcing_red += Forcing[rest[n][0], rest[n][1]]*psi_mean[rest[n][0], rest[n][1]]        
                
                Forcing_blue *= self.delta**2
                Forcing_red *= self.delta**2
                
                self.input_blue = Forcing_blue
                self.input_red = Forcing_red
                self.input = Forcing_blue + Forcing_red
        
        ''' outils '''
        
        def drop_psi(self):
                ''' deletes stream function '''
        
                del self.psi
 
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
