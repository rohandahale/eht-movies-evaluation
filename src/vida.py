######################################################################
# Author: Rohan Dahale, Date: 27 Mar 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import tqdm
import copy
import matplotlib.pyplot as plt
import pdb
import argparse
import os
import glob

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, default='', help='type of model: crescent, ring, disk, edisk, double, point, mring_1_4')
    p.add_argument('--truthcsv', type=str, default='', help='path of truth .csv')
    p.add_argument('--mvcsv',  type=str, default='', help='path of movie .csv')
    p.add_argument('-o', '--outpath', type=str, default='./vida.png',
                   help='name of output file with path')

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

outpath = args.outpath

paths={}
if args.truthcsv!='':
    paths['truth']=args.truthcsv
if args.mvcsv!='':
    paths['recon']=args.mvcsv

######################################################################
colors = {
            'truth'    : 'black',
            'recon'    : 'red',
        }

labels = {
            'truth'     : 'Truth',
            'recon'     : 'Reconstruction',
        }


######################################################################
# Plots
######################################################################

model=args.model

if model=='crescent':
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('Diameter $d (\mu$as)')
    ax[0,0].set_ylim(35,80)

    ax[0,1].set_ylabel('width $w (\mu$as)')
    ax[0,1].set_ylim(10,35)

    ax[1,0].set_ylabel('PA '+ r'$\eta (^{\circ}$ E of N)')
    ax[1,0].set_ylim(-180,180)

    ax[1,1].set_ylabel('Bright. Asym. $A$')
    ax[1,1].set_ylim(0.0,0.6)


    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    for p in paths.keys():
        df = pd.read_csv(paths[p])
        d = 2*df['model_1_r0']/eh.RADPERUAS
        w0 = df['model_1_σ0']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))

        for i in range(len(df['model_1_ξs_1'])):
            if df['model_1_ξs_1'][i]<-np.pi:
                df['model_1_ξs_1'][i] = df['model_1_ξs_1'][i] + 2*np.pi
            if df['model_1_ξs_1'][i]>np.pi:
                df['model_1_ξs_1'][i] = df['model_1_ξs_1'][i] - 2*np.pi

        a = df['model_1_s_1']/2
        n =np.rad2deg(df['model_1_ξs_1'])
        t = df['time']
        
        #for c in [d,w0,a,n]:
        #    for i in range(1,len(c)-1):
        #        if (c==n).all():
        #            j= c[i-1]+ 2*np.pi
        #            k= c[i]+ 2*np.pi
        #            l= c[i+1]+ 2*np.pi
        #            if k> 1.1*j and k> 1.1*l:
        #                c[i] = c[i-1]
        #        else:   
        #            if abs(c[i])> 1.1*abs(c[i-1]) and abs(c[i])> 1.1*abs(c[i+1]):
        #                c[i] = c[i-1]+0.01*c[i-1]
                

        mc=colors[p]
        ax[0,0].plot(t, d,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, w0, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, n,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, a,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)

    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=5.0)

elif model=='ring':
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,3), sharex=True)
     
    alpha = 0.5
    lc='grey'

    ax[0].set_ylabel('Diameter $d (\mu$as)')
    ax[0].set_ylim(35,80)
    ax[1].set_ylabel('width $w (\mu$as)')
    ax[1].set_ylim(10,35)
    ax[0].set_xlabel('Time (UTC)')
    ax[1].set_xlabel('Time (UTC)')


    for i in range(2):
        ax[i].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    for p in paths.keys():
        df = pd.read_csv(paths[p])
        d = 2*df['model_1_r0']/eh.RADPERUAS
        w0 = df['model_1_σ0']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))

        t = df['time']
        
        #for c in [d,w0]:
        #    for i in range(1,len(c)-1):
        #        if abs(c[i])> 1.1*abs(c[i-1]) and abs(c[i])> 1.1*abs(c[i+1]):
        #            c[i] = c[i-1]+0.01*c[i-1]

        mc=colors[p]
        ax[0].plot(t, d,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[1].plot(t, w0, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        
    ax[0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=5.0)

elif model=='disk':
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,3), sharex=True)
     
    alpha = 0.5
    lc='grey'

    ax[0].set_ylabel('Diameter $d (\mu$as)')
    ax[0].set_ylim(35,80)
    ax[1].set_ylabel('Guass FWHM ($\mu$as)')
    ax[1].set_ylim(10,35)
    ax[0].set_xlabel('Time (UTC)')
    ax[1].set_xlabel('Time (UTC)')


    for i in range(2):
        ax[i].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    for p in paths.keys():
        df = pd.read_csv(paths[p])
        d = 2*df['model_1_r0_1']/eh.RADPERUAS
        w0 = df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))

        t = df['time']

        #for c in [d,w0]:
        #    for i in range(1,len(c)-1):
        #        if abs(c[i])> 1.1*abs(c[i-1]) and abs(c[i])> 1.1*abs(c[i+1]):
        #            c[i] = c[i-1]+0.01*c[i-1]
                        
        mc=colors[p]
        ax[0].plot(t, d,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[1].plot(t, w0, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        
    ax[0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=5.0)

elif model=='edisk':
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('Diameter $d (\mu$as)')
    ax[0,0].set_ylim(35,80)

    ax[0,1].set_ylabel('Guass FWHM ($\mu$as)')
    ax[0,1].set_ylim(10,35)

    ax[1,0].set_ylabel('Ellipticity '+r'$\tau$')
    ax[1,0].set_ylim(0.0,0.3)
    
    ax[1,1].set_ylabel(r'$\xi_\tau (^{\circ}$ E of N)')
    ax[1,1].set_ylim(-90,90)

    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    for p in paths.keys():
        df = pd.read_csv(paths[p])
        d = 2*df['model_1_r0_1']/eh.RADPERUAS
        w0 = df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))
        e = abs(df['model_1_τ_1'])

        for i in range(len(df['model_1_ξ_1'])):
            if df['model_1_ξ_1'][i]<-np.pi/2:
                df['model_1_ξ_1'][i] = df['model_1_ξ_1'][i] + np.pi
            if df['model_1_ξ_1'][i]>np.pi/2:
                df['model_1_ξ_1'][i] = df['model_1_ξ_1'][i] - np.pi

        n =np.rad2deg(df['model_1_ξ_1'])
        t = df['time']

        #for c in [d,w0,e,n]:
        #    for i in range(1,len(c)-1):
        #        if (c==n).all():
        #            j= c[i-1]+ np.pi
        #            k= c[i]+ np.pi
        #            l= c[i+1]+ np.pi
        #            if k> 1.1*j and k> 1.1*l:
        #                c[i] = c[i-1]
        #        else:   
        #            if abs(c[i])> 1.1*abs(c[i-1]) and abs(c[i])> 1.1*abs(c[i+1]):
        #                c[i] = c[i-1]+0.01*c[i-1]
                        
        mc=colors[p]
        ax[0,0].plot(t, d,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, w0, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, e,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, n,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)

    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=5.0)

elif model=='double':
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('FWHM 1 ($\mu$as)')
    ax[0,0].set_ylim(0,60)

    ax[0,1].set_ylabel('FWHM 2 ($\mu$as)')
    ax[0,1].set_ylim(0,60)

    ax[1,0].set_ylabel('Separation')
    ax[1,0].set_ylim(40,80)
    
    ax[1,1].set_ylabel(r'PA ($^{\circ}$ E of N)')
    ax[1,1].set_ylim(-180,180)

    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    for p in paths.keys():
        df = pd.read_csv(paths[p])
        
        d1 = np.array(df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        d2 = np.array(df['model_1_σ_2']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        x01 = df['model_1_x0_1']/eh.RADPERUAS
        y01 = df['model_1_y0_1']/eh.RADPERUAS
        x02 = df['model_1_x0_2']/eh.RADPERUAS
        y02 = df['model_1_y0_2']/eh.RADPERUAS
        
        for k in range(len(d1)):
            if d1[k] < d2[k]:
                d1[k] = d2[k]
                d2[k] = d1[k]
                
                x01[k] = x02[k]
                x02[k] = x01[k]
                y01[k] = y02[k]
                y01[k] = y01[k]
        
        pos = np.abs(df['model_1_x0_1']-df['model_1_x0_2'])/eh.RADPERUAS + 1j*np.abs(df['model_1_y0_1']-df['model_1_y0_2'])/eh.RADPERUAS
        r0 = np.abs(pos)
        pa = np.rad2deg(np.angle(pos))
        t = df['time']
        
        #for c in [d1,d2,r0,pa]:
        #    for i in range(1,len(c)-1):
        #        if (c==pa).all():
        #            j= c[i-1]+ 2*np.pi
        #            k= c[i]+ 2*np.pi
        #            l= c[i+1]+ 2*np.pi
        #            if k> 1.1*j and k> 1.1*l:
        #                c[i] = c[i-1]
        #        else:   
        #            if abs(c[i])> 1.1*abs(c[i-1]) and abs(c[i])> 1.1*abs(c[i+1]):
        #                c[i] = c[i-1]+0.01*c[i-1]

        mc=colors[p]
        ax[0,0].plot(t, d1,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, d2,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, r0,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, pa,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)

    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=5.0)

elif model=='point':
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('FWHM 1 ($\mu$as)')
    ax[0,0].set_ylim(0,120)

    ax[0,1].set_ylabel('FWHM 2 ($\mu$as)')
    ax[0,1].set_ylim(0,120)

    ax[1,0].set_ylabel('Separation')
    ax[1,0].set_ylim(-10,50)
    
    ax[1,1].set_ylabel(r'PA ($^{\circ}$ E of N)')
    ax[1,1].set_ylim(-180,180)

    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    for p in paths.keys():
        df = pd.read_csv(paths[p])
        
        d1 = np.array(df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        d2 = np.array(df['model_1_σ_2']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        x01 = df['model_1_x0_1']/eh.RADPERUAS
        y01 = df['model_1_y0_1']/eh.RADPERUAS
        x02 = df['model_1_x0_2']/eh.RADPERUAS
        y02 = df['model_1_y0_2']/eh.RADPERUAS
        
        for k in range(len(d1)):
            if d1[k] < d2[k]:
                d1[k] = d2[k]
                d2[k] = d1[k]
                
                x01[k] = x02[k]
                x02[k] = x01[k]
                y01[k] = y02[k]
                y01[k] = y01[k]
        
        pos = np.abs(df['model_1_x0_1']-df['model_1_x0_2'])/eh.RADPERUAS + 1j*np.abs(df['model_1_y0_1']-df['model_1_y0_2'])/eh.RADPERUAS
        r0 = np.abs(pos)
        pa = np.rad2deg(np.angle(pos))
        t = df['time']

        #for c in [d1,d2,r0,pa]:
        #    for i in range(1,len(c)-1):
        #        if (c==pa).all():
        #            j= c[i-1]+ 2*np.pi
        #            k= c[i]+ 2*np.pi
        #            l= c[i+1]+ 2*np.pi
        #            if k> 1.1*j and k> 1.1*l:
        #                c[i] = c[i-1]
        #        else:   
        #            if abs(c[i])> 1.1*abs(c[i-1]) and abs(c[i])> 1.1*abs(c[i+1]):
        #                c[i] = c[i-1]+0.01*c[i-1]
                        
        mc=colors[p]
        ax[0,0].plot(t, d1,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, d2,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, r0,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, pa,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)

    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=5.0)

else:
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(21,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('Diameter $d (\mu$as)')
    ax[0,0].set_ylim(35,80)

    ax[0,1].set_ylabel('width $w (\mu$as)')
    ax[0,1].set_ylim(10,35)

    ax[0,2].set_ylabel('PA '+ r'$\eta (^{\circ}$ E of N)')
    ax[0,2].set_ylim(-180,180)

    ax[1,0].set_ylabel('Bright. Asym. $A$')
    ax[1,0].set_ylim(0.0,0.55)

    ax[1,1].set_ylabel('Ellipticity '+r'$\tau$')
    ax[1,1].set_ylim(0.0,0.3)

    ax[1,2].set_ylabel('2nd PA '+ r'$\eta_2 (^{\circ}$ E of N)')
    ax[1,2].set_ylim(-180,180)


    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')
    ax[1,2].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(3):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################

    for p in paths.keys():
    
        df = pd.read_csv(paths[p])
        d = 2*df['model_1_r0']/eh.RADPERUAS
        w0 = df['model_1_σ0']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))
        e = abs(df['model_1_τ'])
        for i in range(len(df['model_1_ξs_1'])):
            if df['model_1_ξs_1'][i]<-np.pi:
                df['model_1_ξs_1'][i] = df['model_1_ξs_1'][i] + 2*np.pi
            if df['model_1_ξs_1'][i]>np.pi:
                df['model_1_ξs_1'][i] = df['model_1_ξs_1'][i] - 2*np.pi
        for i in range(len(df['model_1_ξs_2'])):
            if df['model_1_ξs_2'][i]<-np.pi:
                df['model_1_ξs_2'][i] = df['model_1_ξs_2'][i] + 2*np.pi
            if df['model_1_ξs_2'][i]>np.pi:
                df['model_1_ξs_2'][i] = df['model_1_ξs_2'][i] - 2*np.pi
        a = df['model_1_s_1']/2
        n =np.rad2deg(df['model_1_ξs_1'])
        n2 =np.rad2deg(df['model_1_ξs_2'])
        t = df['time']
        mc=colors[p]
        
        #for c in [d,w0,a,e,n,n2]:
        #    for i in range(1,len(c)-1):
        #        if (c==n).all() or (c==n2).all():
        #            j= c[i-1]+ 2*np.pi
        #            k= c[i]+ 2*np.pi
        #            l= c[i+1]+ 2*np.pi
        #            if k> 1.1*j and k> 1.1*l:
        #                c[i] = c[i-1]
        #        else:   
        #            if abs(c[i])> 1.1*abs(c[i-1]) and abs(c[i])> 1.1*abs(c[i+1]):
        #                c[i] = c[i-1]+0.01*c[i-1]
                        
        ax[0,0].plot(t, d,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, w0, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[0,2].plot(t, n,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, a,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, e,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)    
        ax[1,2].plot(t, n2, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1, color=lc, alpha=alpha)

    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(3.1, 1.4), markerscale=5.0)
    
#else:
    #print('Model not in the list of plot functions')
    
plt.savefig(outpath, bbox_inches='tight', dpi=300)
print(f'{os.path.basename(outpath)} is created')