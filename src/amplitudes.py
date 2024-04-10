######################################################################
# Author: Ilje Cho, Rohan Dahale, Date: 14 Mar 2024
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
    p.add_argument('--mv', type=str, default='', help='path of .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./amp.png', 
                   help='name of output file with path')
    p.add_argument('--pol',  type=str, default='I',help='I,Q,U,V')
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
plt.style.use('dark_background')

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

pathmov  = args.mv
outpath = args.outpath
pol = args.pol

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

if pol=='I':
   if len(im.ivec)>0:
        polpath=pathmov
   else:
       print('Parse a vaild pol value')
       exit()
elif pol=='Q':
    if len(im.qvec)>0:
        polpath=pathmov
    else:
        print('Parse a vaild pol value')
        exit()
elif pol=='U':
    if len(im.uvec)>0:
        polpath=pathmov
    else:
        print('Parse a vaild pol value')
        exit()
elif pol=='V':
    if len(im.vvec)>0:
        polpath=pathmov
    else:
        print('Parse a vaild pol value')
        exit()
else:
    print('Parse a vaild pol value')
    
color = 'darkorange'
label = 'Reconstruction'


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


bs_list = [('AZ', 'LM'), ('AA', 'AZ'), ('LM', 'SM')]


######################################################################

mv = eh.movie.load_hdf5(polpath)
    
amp_mod_time = dict()
amp_mod_win = dict()

for bs in bs_list:
    amp_time, amp_window = [], []
    for ii in range(len(times)):
        tstamp = times[ii]
        im = mv.get_image(times[ii])
        im.rf = obslist_t[ii].rf
        if im.xdim%2 == 1:
            im = im.regrid_image(targetfov=im.fovx(), npix=im.xdim-1)
        obs_mod = im.observe_same(obslist_t[ii], add_th_noise=False, ttype='fast')
        amp_mod = pd.DataFrame(obs_mod.data)
            
        # select baseline
        subtab  = select_baseline(amp_mod, bs[0], bs[1])
        try:
            idx = np.where(np.round(subtab['time'].values,3)  == np.round(tstamp,3))[0][0]                
            amp_time.append(subtab['time'][idx])                
            if args.pol=='I': amp_window.append(abs(subtab['vis'][idx]))
            if args.pol=='Q': amp_window.append(abs(subtab['qvis'][idx]))
            if args.pol=='U': amp_window.append(abs(subtab['uvis'][idx]))
            if args.pol=='V': amp_window.append(abs(subtab['vvis'][idx]))
        except:
            pass

    amp_mod_time[bs] = amp_time
    amp_mod_win[bs] = amp_window
    
mov_amp = [amp_mod_time, amp_mod_win]
######################################################################

vtab = copy(amp)
fix_xax = True

numplt = 3
xmin = min(vtab['time'].values)
xmax = max(vtab['time'].values)

if args.pol=='I': vis='vis'
if args.pol=='Q': vis='qvis'
if args.pol=='U': vis='uvis'
if args.pol=='V': vis='vvis'

if args.pol=='I': sigma='sigma'
if args.pol=='Q': sigma='qsigma'
if args.pol=='U': sigma='usigma'
if args.pol=='V': sigma='vsigma'
                
fig = plt.figure(figsize=(21,4))
fig.subplots_adjust(wspace=0.3)

axs = []
for i in range(1,numplt+1):
    axs.append(fig.add_subplot(1,3,i))

for i in tqdm(range(numplt)):
    subtab  = select_baseline(vtab, bs_list[i][0], bs_list[i][1])
    axs[i].errorbar(subtab['time'], abs(subtab[vis]), yerr=subtab[sigma],
                    c='white', mec='white', marker='o', ls="None", ms=5, alpha=0.5, label='Data')

    mv = eh.movie.load_hdf5(polpath)
    amp_mod_time, amp_mod_win = mov_amp
    axs[i].errorbar(amp_mod_time[bs_list[i]], amp_mod_win[bs_list[i]], 
                    c=color, marker='o', ms=2.5, ls="none", label=label, alpha=0.5)
    

    axs[i].axhline(y=0, c='k', ls=':')
    axs[i].set_title("%s-%s" %(bs_list[i][0], bs_list[i][1]), fontsize=18)

    if i == 0:
        axs[i].legend(ncols=6, loc='best',  bbox_to_anchor=(2.3, 1.3), markerscale=5.0)
    axs[i].set_xlabel('Time (UTC)')
    axs[i].set_ylabel(f'Amp: Stokes {pol}')

    if fix_xax:
        axs[i].set_xlim(xmin-0.5, xmax+0.5)

fig.subplots_adjust(top=0.93)
plt.savefig(args.outpath, bbox_inches='tight', dpi=300)