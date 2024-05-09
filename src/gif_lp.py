import ehtim as eh
import ehtplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pdb
import argparse
import os
import glob

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='', help='path of truth .hdf5')
    p.add_argument('--mv', type=str, default='', help='path of .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./gif.gif', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='sct, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

######################################################################
# Plotting Setup
######################################################################
#plt.rc('text', usetex=True)
import matplotlib as mpl
#mpl.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'monospace': ['Computer Modern Typewriter']})
mpl.rcParams['figure.dpi']=300
#mpl.rcParams["mathtext.default"] = 'regular'
#plt.style.use('dark_background')
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18

from matplotlib import font_manager
font_dirs = font_manager.findSystemFonts(fontpaths='./fonts/', fontext="ttf")
#mpl.rc('text', usetex=True)

fe = font_manager.FontEntry(
    fname='./fonts/Helvetica.ttf',
    name='Helvetica')
font_manager.fontManager.ttflist.insert(0, fe) # or append is fine
mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
######################################################################

# Time average data to 60s
obs = eh.obsdata.load_uvfits(args.data)
obs = obs.avg_coherent(60.0)

# From GYZ: If data used by pipelines is descattered (refractive + diffractive),
# Add 2% error and deblur original data.
if args.scat=='dsct':
    obs = obs.add_fractional_noise(0.02)
    import ehtim.scattering.stochastic_optics as so
    sm = so.ScatteringModel()
    obs = sm.Deblur_obs(obs)

obs.add_scans()
times = []
for t in obs.scans:
    times.append(t[0])
obslist = obs.split_obs()
######################################################################
outpath = args.outpath


paths={}

if args.truthmv!='':
    paths['truth']=args.truthmv
if args.mv!='':
    paths['recon']=args.mv


# Truncating the times and obslist based on submitted movies
obslist_tn=[]
min_arr=[] 
max_arr=[]
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    min_arr.append(min(mv.times))
    max_arr.append(max(mv.times))
x=np.argwhere(times>max(min_arr))
ntimes=[]
for t in x:
    ntimes.append(times[t[0]])
    obslist_tn.append(obslist[t[0]])
times=[]
obslist_t=[]
y=np.argwhere(min(max_arr)>ntimes)
for t in y:
    times.append(ntimes[t[0]])
    obslist_t.append(obslist_tn[t[0]])
    
######################################################################
# Set parameters
npix   = 128
fov    = 120 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

titles = {  
            'truth'       : 'Truth',
            'recon'       : 'Reconstruction',
        }

######################################################################
# Adding times where there are gaps and assigning cmap as binary_us in the gaps
dt=[]
for i in range(len(times)-1):
    dt.append(times[i+1]-times[i])
    
mean_dt=np.mean(np.array(dt))

u_times=[]
cmapsl = []
for i in range(len(times)-1):
    if times[i+1]-times[i] > mean_dt:
        j=0
        while u_times[len(u_times)-1] < times[i+1]-mean_dt:
            u_times.append(times[i]+j*mean_dt)
            cmapsl.append('binary_usr')
            j=j+1
    else:
        u_times.append(times[i])
        cmapsl.append('binary_usr')

######################################################################

imlistIs = {}
for p in paths.keys():
    mov = eh.movie.load_hdf5(paths[p])
    imlistI = []
    for t in u_times:
        im = mov.get_image(t)
        #if p=='truth':
        #    im = im.blur_circ(fwhm_i=15*eh.RADPERUAS).regrid_image(fov, npix)
        #else:
        im = im.blur_circ(fwhm_i=blur).regrid_image(fov, npix)
        #im.ivec=im.ivec/im.total_flux()
        imlistI.append(im)
    imlistIs[p] =imlistI
    #med = np.median(imlistIs[p],axis=0)
    #for i in range(len(imlistIs[p])):
        #imlistIs[p][i]= np.clip(imlistIs[p][i]-med,0,1)
        

def writegif(movieIs, titles, paths, outpath='./', fov=None, times=[], cmaps=cmapsl, interp='gaussian', fps=20):

    if len(paths.keys())!=1:
        fig, ax = plt.subplots(nrows=1, ncols=len(paths.keys()), figsize=(8,5))
        #fig.tight_layout()
        fig.subplots_adjust(hspace=0.01, wspace=0.05, top=0.8, bottom=0.01, left=0.01, right=0.8)
        # Set axis limits
        lims = None
        if fov:
            fov  = fov / eh.RADPERUAS
            lims = [fov//2, -fov//2, -fov//2, fov//2]

        # Set colorbar limits
        TBfactor = 3.254e13/(movieIs['recon'][0].rf**2 * movieIs['recon'][0].psize**2)/1e9    
        vmax, vmin = max(movieIs['recon'][0].ivec)*TBfactor, min(movieIs['recon'][0].ivec)*TBfactor

        polmovies={}
        for i, p in enumerate(movieIs.keys()):
            if len(movieIs[p][0].qvec) and len(movieIs[p][0].uvec) > 0:
                polmovies[p]=movieIs[p]

        def plot_frame(f):
            for i, p in enumerate(movieIs.keys()):
                ax[i].clear() 
                TBfactor = 3.254e13/(movieIs[p][f].rf**2 * movieIs[p][f].psize**2)/1e9
                im =ax[i].imshow(np.array(movieIs[p][f].imarr(pol='I'))*TBfactor, cmap=cmaps[f], interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

                ax[i].set_title(titles[p], fontsize=18)
                ax[i].set_xticks([]), ax[i].set_yticks([])

                if p in polmovies.keys():
                    self = polmovies[p][f]
                    amp = np.sqrt(self.qvec**2 + self.uvec**2)
                    scal=np.max(amp)*0.5

                    vx = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)
                    vy = ( np.cos(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)

                    # tick color will be proportional to mfrac
                    mfrac=(amp/self.ivec).reshape(self.xdim, self.ydim)

                    pcut = 0.1
                    mcut = 0.
                    skip = 10
                    imarr = self.imvec.reshape(self.ydim, self.xdim)
                    Imax=max(self.imvec)
                    mfrac = np.ma.masked_where(imarr < pcut * Imax, mfrac) 

                    #new version with sharper cuts
                    mfrac_map=(np.sqrt(self.qvec**2+self.uvec**2)).reshape(self.xdim, self.ydim)
                    QUmax=max(np.sqrt(self.qvec**2+self.uvec**2))
                    pcut=0.1
                    mfrac_m = np.ma.masked_where(mfrac_map < pcut * QUmax , mfrac)
                    pcut=0.1
                    mfrac_m = np.ma.masked_where(imarr < pcut * Imax, mfrac_m)
                    ######

                    pixel=self.psize/eh.RADPERUAS #uas
                    FOV=pixel*self.xdim

                    # generate 2 2d grids for the x & y bounds
                    y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                                    slice(-FOV/2, FOV/2, pixel)]

                    x = np.ma.masked_where(imarr < pcut * Imax, x) 
                    y = np.ma.masked_where(imarr < pcut * Imax, y) 
                    vx = np.ma.masked_where(imarr < pcut * Imax, vx) 
                    vy = np.ma.masked_where(imarr < pcut * Imax, vy) 

    
                    cnorm=Normalize(vmin=0.0, vmax=1.0)
                    tickplot = ax[i].quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
                                   mfrac_m[::skip,::skip],
                                   headlength=0,
                                   headwidth = 1,
                                   pivot='mid',
                                   width=0.01,
                                   cmap='rainbow',
                                   norm=cnorm,
                                   scale=16)
            if f==0:
                ax1 = fig.add_axes([0.82, 0.1, 0.02, 0.6] , anchor = 'E') 
                cbar = fig.colorbar(tickplot, cmap='rainbow', cax=ax1, pad=0.14,fraction=0.038, orientation="vertical") 
                cbar.set_label('$|m|$') 

            plt.suptitle(f"{u_times[f]:.2f} UT", y=0.95, fontsize=22)

            return fig
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5,5))
        #fig.tight_layout()
        # Set axis limits
        lims = None
        if fov:
            fov  = fov / eh.RADPERUAS
            lims = [fov//2, -fov//2, -fov//2, fov//2]

        # Set colorbar limits
        TBfactor = 3.254e13/(movieIs['recon'][0].rf**2 * movieIs['recon'][0].psize**2)/1e9    
        vmax, vmin = max(movieIs['recon'][0].ivec)*TBfactor, min(movieIs['recon'][0].ivec)*TBfactor

        polmovies={}
        for i, p in enumerate(movieIs.keys()):
            if len(movieIs[p][0].qvec) and len(movieIs[p][0].uvec) > 0:
                polmovies[p]=movieIs[p]

        def plot_frame(f):
            for i, p in enumerate(movieIs.keys()):
                ax.clear() 
                TBfactor = 3.254e13/(movieIs[p][f].rf**2 * movieIs[p][f].psize**2)/1e9
                im =ax.imshow(np.array(movieIs[p][f].imarr(pol='I'))*TBfactor, cmap=cmaps[f], interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

                ax.set_title(f"{u_times[f]:.2f} UT", fontsize=18)
                ax.set_xticks([]), ax.set_yticks([])

                if p in polmovies.keys():
                    self = polmovies[p][f]
                    amp = np.sqrt(self.qvec**2 + self.uvec**2)
                    scal=np.max(amp)*0.5

                    vx = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)
                    vy = ( np.cos(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)

                    # tick color will be proportional to mfrac
                    mfrac=(amp/self.ivec).reshape(self.xdim, self.ydim)

                    pcut = 0.1
                    mcut = 0.
                    skip = 10
                    imarr = self.imvec.reshape(self.ydim, self.xdim)
                    Imax=max(self.imvec)
                    mfrac = np.ma.masked_where(imarr < pcut * Imax, mfrac) 

                    #new version with sharper cuts
                    mfrac_map=(np.sqrt(self.qvec**2+self.uvec**2)).reshape(self.xdim, self.ydim)
                    QUmax=max(np.sqrt(self.qvec**2+self.uvec**2))
                    pcut=0.1
                    mfrac_m = np.ma.masked_where(mfrac_map < pcut * QUmax , mfrac)
                    pcut=0.1
                    mfrac_m = np.ma.masked_where(imarr < pcut * Imax, mfrac_m)
                    ######

                    pixel=self.psize/eh.RADPERUAS #uas
                    FOV=pixel*self.xdim

                    # generate 2 2d grids for the x & y bounds
                    y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                                    slice(-FOV/2, FOV/2, pixel)]

                    x  = np.ma.masked_where(imarr < pcut * Imax, x) 
                    y  = np.ma.masked_where(imarr < pcut * Imax, y) 
                    vx = np.ma.masked_where(imarr < pcut * Imax, vx) 
                    vy = np.ma.masked_where(imarr < pcut * Imax, vy) 

    
                    cnorm=Normalize(vmin=0.0, vmax=1.0)
                    tickplot = ax.quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
                                   mfrac_m[::skip,::skip],
                                   headlength=0,
                                   headwidth = 1,
                                   pivot='mid',
                                   width=0.01,
                                   cmap='rainbow',
                                   norm=cnorm,
                                   scale=16)
            if f==0:
                ax1 = fig.add_axes([0.825, 0.11, 0.035, 0.77] , anchor = 'E') 
                cbar = fig.colorbar(tickplot, cmap='rainbow', cax=ax1, pad=0.14,fraction=0.038, orientation="vertical") 
                cbar.set_label('$|m|$') 

            return fig
    
    def update(f):
        return plot_frame(f)

    ani = animation.FuncAnimation(fig, update, frames=len(u_times), interval=1e3/fps)
    wri = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)

    # Save gif
    ani.save(outpath, writer=wri, dpi=100)

writegif(imlistIs, titles, paths, outpath=outpath, fov=fov, times=u_times, cmaps=cmapsl)
