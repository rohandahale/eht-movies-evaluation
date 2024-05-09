######################################################################
# Author: Rohan Dahale, Date: 9 May 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

import argparse
import os
import glob
from tqdm import tqdm
import itertools 
import sys
from copy import copy

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='', help='path of truth .hdf5')
    p.add_argument('--mv', type=str, default='', help='path of .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./amp.png', 
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
plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
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


amp = pd.DataFrame(obs.data)

obs.add_scans()
times = []
for t in obs.scans:
    times.append(t[0])
obslist = obs.split_obs()
######################################################################
outpath = args.outpath
    
if args.truthmv!='':
    pathmovt=args.truthmv
if args.mv!='':
    pathmov=args.mv
    
labels={
        'truth' : 'Truth',
        'recon' :  'Reconstruction'   
}

colors={
        'truth': 'black',
        'recon': 'red'
}


def select_baseline(tab, st1, st2):
        stalist = list(itertools.permutations([st1, st2]))
        idx = []
        for stations in stalist:
            ant1, ant2 = stations
            subidx = np.where((tab["t1"].values == ant1) &
                              (tab["t2"].values == ant2) )
            idx +=  list(subidx[0])
    
        newtab = tab.take(idx).sort_values(by=["time"]).reset_index(drop=True)
        return newtab
    

######################################################################
# Truncating the times and obslist based on submitted movies
obslist_tn=[]
min_arr=[] 
max_arr=[]
mv=eh.movie.load_hdf5(pathmov)
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

mv=eh.movie.load_hdf5(pathmov)
im=mv.get_image(times[0])
if len(im.ivec)>0 and len(im.qvec)>0 and len(im.uvec)>0:
    polpath=pathmov
else:
    print('There is no I,Q or U')
    exit()
######################################################################
mv = eh.movie.load_hdf5(polpath)
mb_time, mb_window = [], []
for ii in range(len(times)):
    tstamp = times[ii]
    im = mv.get_image(times[ii])
    im.rf = obslist_t[ii].rf
    if im.xdim%2 == 1:
        im = im.regrid_image(targetfov=im.fovx(), npix=im.xdim-1)
    obs_mod = im.observe_same(obslist_t[ii], add_th_noise=False, ttype='fast')
    amp_mod = pd.DataFrame(obs_mod.data)
    # select baseline
    subtab  = select_baseline(amp_mod, 'AA', 'AZ')
    try:
        idx = np.where(np.round(subtab['time'].values,3)  == np.round(tstamp,3))[0][0]                
        mb_time.append(subtab['time'][idx]) 
        mb_window.append(abs(subtab['qvis'][idx]+1j*subtab['uvis'][idx]/subtab['vis'][idx]))  
    except:
        pass
    
######################################################################
subtab  = select_baseline(amp, 'AA', 'AZ')
mbreve = abs(subtab['qvis']+1j*subtab['uvis']/subtab['vis'])
mbreve_sig = np.sqrt((mbreve**2)*(subtab['qsigma']**2-subtab['usigma']**2+(subtab['sigma']**2/abs(subtab['vis'])**2)))/abs(subtab['vis'])
plt.errorbar(mb_time, mb_window, c='red', marker='o', ms=2.5, ls="none", label='Reconstruction', alpha=0.5)

if args.truthmv=='':
    plt.errorbar(subtab['time'], mbreve, yerr=mbreve_sig, c='black', mec='black', marker='o', ls="None", ms=5, alpha=0.5, label='AA-AZ')

if args.truthmv!='':
    # Time average data to 60s
    del obs
    del obslist
    obs = eh.obsdata.load_uvfits(args.data)
    obs = obs.avg_coherent(60.0)

    # From GYZ: If data used by pipelines is descattered (refractive + diffractive),
    # Add 2% error and deblur original data.
    if args.scat=='dsct':
        obs = obs.add_fractional_noise(0.02)
        import ehtim.scattering.stochastic_optics as so
        sm = so.ScatteringModel()
        obs = sm.Deblur_obs(obs)


    amp = pd.DataFrame(obs.data)

    obs.add_scans()
    times = []
    for t in obs.scans:
        times.append(t[0])
    obslist = obs.split_obs()

    ######################################################################
    # Truncating the times and obslist based on submitted movies
    del obslist_tn
    del min_arr
    del max_arr
    
    obslist_tn=[]
    min_arr=[] 
    max_arr=[]
    del mv
    mv=eh.movie.load_hdf5(pathmovt)
    min_arr.append(min(mv.times))
    max_arr.append(max(mv.times))
    x=np.argwhere(times>max(min_arr))
    del ntimes
    ntimes=[]
    for t in x:
        ntimes.append(times[t[0]])
        obslist_tn.append(obslist[t[0]])
    del times
    times=[]
    del obslist_t
    obslist_t=[]
    del y
    y=np.argwhere(min(max_arr)>ntimes)
    for t in y:
        times.append(ntimes[t[0]])
        obslist_t.append(obslist_tn[t[0]])
    ######################################################################
    del mv
    mv=eh.movie.load_hdf5(pathmovt)
    del im
    im=mv.get_image(times[0])
    del polpath
    if len(im.ivec)>0 and len(im.qvec)>0 and len(im.uvec)>0:
        polpath=pathmovt
    else:
        print('There is no I,Q or U')
        exit()
    ######################################################################
    del mv
    mv = eh.movie.load_hdf5(polpath)
    del mb_time
    del mb_window
    mb_time, mb_window = [], []
    del tstamp
    del im
    del obs_mod
    del amp_mod
    del subtab
    del idx
    
    for ii in range(len(times)):
        tstamp = times[ii]
        im = mv.get_image(times[ii])
        im.rf = obslist_t[ii].rf
        if im.xdim%2 == 1:
            im = im.regrid_image(targetfov=im.fovx(), npix=im.xdim-1)
        obs_mod = im.observe_same(obslist_t[ii], add_th_noise=False, ttype='fast')
        amp_mod = pd.DataFrame(obs_mod.data)
        # select baseline
        subtab  = select_baseline(amp_mod, 'AA', 'AZ')
        try:
            idx = np.where(np.round(subtab['time'].values,3)  == np.round(tstamp,3))[0][0]                
            mb_time.append(subtab['time'][idx]) 
            mb_window.append(abs(subtab['qvis'][idx]+1j*subtab['uvis'][idx]/subtab['vis'][idx]))  
        except:
            pass
        
    ######################################################################
    plt.errorbar(mb_time, mb_window, c='black', marker='o', ms=2.5, ls="none", label='Truth', alpha=0.5, zorder=0)

  
plt.yscale('log')
plt.ylim(0.01,1)
plt.xlabel('Time (UTC)')
plt.ylabel('$|\\breve{m}|$')
plt.legend(ncols=3, loc='best',  bbox_to_anchor=(1, 1.2), markerscale=5.0)
plt.savefig(args.outpath, bbox_inches='tight', dpi=300)