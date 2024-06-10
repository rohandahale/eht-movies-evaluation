######################################################################
# Author: Rohan Dahale, Date: 25 Mar 2024
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
plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
plt.rcParams["ytick.major.size"]=5
plt.rcParams["ytick.minor.size"]=2.5
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
obs.add_scans()
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
    
pathmov  = args.mv
pathmovt  = args.truthmv
outpath = args.outpath

path = pathmov

######################################################################

# Truncating the times and obslist based on submitted movies
obslist_tn=[]
min_arr=[] 
max_arr=[]

paths=[pathmovt, pathmov]
for x in paths:
    mv=eh.movie.load_hdf5(x)
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

color = 'red'

label = 'Reconstruction'

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(28,8), sharex=True)

ax[0].set_ylabel('nxcorr (I)')
ax[1].set_ylabel('nxcorr (Q)')
ax[2].set_ylabel('nxcorr (U)')
ax[3].set_ylabel('nxcorr (V)')

ax[0].set_xlabel('Time (UTC)')
ax[1].set_xlabel('Time (UTC)')
ax[2].set_xlabel('Time (UTC)')
ax[3].set_xlabel('Time (UTC)')


ax[0].set_ylim(0.5,1.5)
ax[1].set_ylim(0.5,1.5)
ax[2].set_ylim(0.5,1.5)
ax[3].set_ylim(0.5,1.5)


mvt=eh.movie.load_hdf5(pathmovt)
if args.scat=='dsct':
    mvt=mvt.blur_circ(fwhm_x=15*eh.RADPERUAS, fwhm_x_pol=15*eh.RADPERUAS, fwhm_t=0)

table_vals = {} #pd.DataFrame()
table_vals['I'] = []
table_vals['Q'] = []
table_vals['U'] = []
table_vals['V'] = []
        
pollist=['I','Q','U','V']
k=0
for pol in pollist:
    mv=eh.movie.load_hdf5(path)
    im=mv.im_list()[0]
    
    if pol=='I':
        if len(im.ivec)>0:
            polpath=path
        else:
            polpath=''
            table_vals['I'].append('-')
    elif pol=='Q':
        if len(im.qvec)>0:
            polpath=path
        else:
            polpath=''
            table_vals['Q'].append('-')
    elif pol=='U':
        if len(im.uvec)>0:
            polpath=path
        else:
            polpath=''
            table_vals['U'].append('-')
    elif pol=='V':
        if len(im.vvec)>0:
            polpath=path
        else:
            polpath=''
            table_vals['V'].append('-')
    else:
        print('Parse a vaild pol value')

    if polpath!='':
        mv=eh.movie.load_hdf5(polpath)

        imlist = [mv.get_image(t) for t in times]
        imlistarr=[]
        for im in imlist:
            im.ivec=im.ivec/im.total_flux()
            imlistarr.append(im.imarr(pol=pol))
        mean = np.mean(imlistarr,axis=0)
        for im in imlist:
            if pol=='I':
                im.ivec= mean.flatten()
            elif pol=='Q':
                im.qvec= mean.flatten()
            elif pol=='U':
                im.uvec= mean.flatten()
            elif pol=='V':
                im.vvec= mean.flatten()
    

        imlist_t =[mvt.get_image(t) for t in times]
        imlistarr=[]
        for im in imlist_t:
            im.ivec=im.ivec/im.total_flux()
            imlistarr.append(im.imarr(pol=pol))
        mean = np.mean(imlistarr,axis=0)
        for im in imlist_t:
            if pol=='I':
                im.ivec= mean.flatten()
            elif pol=='Q':
                im.qvec= mean.flatten()
            elif pol=='U':
                im.uvec= mean.flatten()
            elif pol=='V':
                im.vvec= mean.flatten()

        nxcorr_t=[]
        nxcorr_tab=[]
        
        i=0
        for im in imlist:
            nxcorr=imlist_t[i].compare_images(im, pol=pol, metric=['nxcorr'])
            nxcorr_t.append(nxcorr[0][0])
            nxcorr_tab.append(nxcorr[0][0])
            i=i+1

        table_vals[pol].append(np.round(np.mean(np.array(nxcorr_tab)),3))
        
        mc=color
        alpha = 0.5
        lc=color

        if k==0:
            ax[k].plot(times, nxcorr_t,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha, label=label)
        else:
            ax[k].plot(times, nxcorr_t,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha)  
    
    ax[k].hlines(1, xmin=10.5, xmax=14.5, color='black', ls='--', lw=1.5, zorder=0)
    #ax[k].yaxis.set_ticklabels([])
    k=k+1

df = pd.DataFrame.from_dict(table_vals)
df.rename(index={0:'nxcorr'},inplace=True)

table = ax[1].table(cellText=df.values,
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='bottom',
                    bbox=[0.5, -0.4, 1.5, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(22)
for c in table.get_children():
    c.set_edgecolor('none')
    c.set_text_props(color='black')
    c.set_facecolor('none')
    c.set_edgecolor('black')
#ax[0].legend(ncols=2, loc='best',  bbox_to_anchor=(2.1, 1.2), markerscale=5.0)

plt.savefig(outpath, bbox_inches='tight', dpi=300)
