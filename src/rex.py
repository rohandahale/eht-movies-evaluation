######################################################################
# Author: Kotaro Moriyama, Date: 25 Mar 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
from ehtim.const_def import *
import tqdm
from scipy import interpolate, optimize, stats
from scipy.interpolate import RectBivariateSpline
import copy
from astropy.constants import k_B,c
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import itertools

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
    return p

# List of parsed arguments
args = create_parser().parse_args()

######################################################################
# REx functions
######################################################################
def extract_ring_quantites(image,xc=None,yc=None, rcutoff=5):
    Npa=360
    Nr=100

    if xc==None or yc==None:
    # Compute the image center -----------------------------------------------------------------------------------------
        xc,yc = fit_ring(image)
    # Gridding and interpolation ---------------------------------------------------------------------------------------
    x= np.arange(image.xdim)*image.psize/RADPERUAS
    y= np.flip(np.arange(image.ydim)*image.psize/RADPERUAS)
    z = image.imarr()
    f_image = interpolate.interp2d(x,y,z,kind="cubic") # init interpolator
    #f_image = RectBivariateSpline(x, y, z)

    # Create a mesh grid in polar coordinates
    radial_imarr = np.zeros([Nr,Npa])

    pa = np.linspace(0,360,Npa)
    pa_rad = np.deg2rad(pa)
    radial = np.linspace(0,50,Nr)
    dr = radial[-1]-radial[-2]

    Rmesh, PAradmesh = np.meshgrid(radial, pa_rad)
    x = Rmesh*np.sin(PAradmesh) + xc
    y = Rmesh*np.cos(PAradmesh) + yc
    for r in range(Nr):
        z = [f_image(x[i][r],y[i][r]) for i in range(len(pa))]
        radial_imarr[r,:] = np.array(z)[:,0]
    radial_imarr = np.fliplr(radial_imarr)
    # Calculate the r_pk at each PA and average -> using radius  --------------------------------------------------------
    # Caluculating the ring width from rmin and rmax
    peakpos = np.unravel_index(np.argmax(radial_imarr), shape=radial_imarr.shape)

    Rpeak=[]
    Rmin=[]
    Rmax=[]
    ridx_r50= np.argmin(np.abs(radial - 50))
    I_floor = radial_imarr[ridx_r50,:].mean()
    for ipa in range(len(pa)):
        tmpIr = copy.copy(radial_imarr[:,ipa])-I_floor
        tmpIr[np.where(radial < rcutoff)]=0
        ridx_pk = np.argmax(tmpIr)
        rpeak = radial[ridx_pk]
        if ridx_pk > 0 and ridx_pk < Nr-1:
            val_list= tmpIr[ridx_pk-1:ridx_pk+2]
            rpeak = quad_interp_radius(rpeak, dr, val_list)[0]
        idx = np.array(np.where(tmpIr > tmpIr.max()/2.0))
        Rpeak.append(rpeak)
        # if tmpIr < 0, make rmin & rmax nan
        rmin,rmax = calc_width(tmpIr,radial,rpeak)
        # append
        Rmin.append(rmin)
        Rmax.append(rmax)
    paprofile = pd.DataFrame()
    paprofile["PA"] = pa
    paprofile["rpeak"] = Rpeak
    paprofile["rhalf_max"]=Rmax
    paprofile["rhalf_min"]=Rmin

    D = np.mean(paprofile["rpeak"]) * 2
    Derr = paprofile["rpeak"].std() * 2
    W = np.mean(paprofile["rhalf_max"] - paprofile["rhalf_min"])
    Werr =  (paprofile["rhalf_max"] - paprofile["rhalf_min"]).std()

    # Caluculate the orienttion angle, contrast, and assymetry
    rin  = D/2.-W/2.
    rout  = D/2.+W/2.
    if rin <= 0.:
        rin  = 0.

    exptheta =np.exp(1j*pa_rad)

    pa_ori_r=[]
    amp_r = []
    ridx1 = np.argmin(np.abs(radial - rin))
    ridx2 = np.argmin(np.abs(radial - rout))
    for r in range(ridx1, ridx2+1, 1):
        amp =  (radial_imarr[r,:]*exptheta).sum()/(radial_imarr[r,:]).sum()
        amp_r.append(amp)
        pa_ori = np.angle(amp, deg=True)
        pa_ori_r.append(pa_ori)
    pa_ori_r=np.array(pa_ori_r)
    amp_r = np.array(amp_r)
    PAori = stats.circmean(pa_ori_r,high=360,low=0)
    PAerr = stats.circstd(pa_ori_r,high=360,low=0)
    A = np.mean(np.abs(amp_r))
    Aerr = np.std(np.abs(amp_r))

    ridx_r5= np.argmin(np.abs(radial - 5))
    ridx_pk = np.argmin(np.abs(radial - D/2))
    fc = radial_imarr[0:ridx_r5,:].mean()/radial_imarr[ridx_pk,:].mean()

    # source size from 2nd moment
    fwhm_maj,fwhm_min,theta = image.fit_gauss()
    fwhm_maj /= RADPERUAS
    fwhm_min /= RADPERUAS


    # calculate flux ratio
    Nxc = int(xc/image.psize*RADPERUAS)
    Nyc = int(yc/image.psize*RADPERUAS)
    hole = extract_hole(image,Nxc,Nyc,r=rin)
    ring = extract_ring(image,Nxc,Nyc,rin=rin, rout=rout)
    outer = extract_outer(image,Nxc,Nyc,r=rout)
    hole_flux = hole.total_flux()
    outer_flux = outer.total_flux()
    ring_flux = ring.total_flux()

    Shole  = np.pi*rin**2
    Souter = (2.*rout)**2.-np.pi*rout**2
    Sring = np.pi*rout**2-np.pi*rin**2

    # convert uas^2 to rad^2
    Shole = Shole*RADPERUAS**2
    Souter = Souter*RADPERUAS**2
    Sring = Sring*RADPERUAS**2

    #unit K brighthness temperature
    freq = image.rf*u.Hz
    hole_dflux  = hole_flux/Shole*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    outer_dflux = outer_flux/Souter*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    ring_dflux = ring_flux/Sring*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value

    # output dictionary
    outputs = dict(
        time_utc = image.time,
        radial_imarr=radial_imarr,
        peak_idx=peakpos,
        rpeak=radial[peakpos[0]],
        papeak=pa[peakpos[1]],
        paprofile=paprofile,
        xc=xc,
        yc=yc,
        r = radial,
        PAori = PAori,
        PAerr = PAerr,
        A = A,
        Aerr = Aerr,
        fc = fc,
        D = D,
        Derr = Derr,
        W = W,
        Werr = Werr,
        fwhm_maj=fwhm_maj,
        fwhm_min=fwhm_min,
        hole_flux = hole_flux,
        outer_flux = outer_flux,
        ring_flux = ring_flux,
        totalflux = image.total_flux(),
        hole_dflux = hole_dflux,
        outer_dflux = outer_dflux,
        ring_dflux = ring_dflux
    )
    return outputs

# Clear ring structures
def extract_hole(image,Nxc,Nyc, r=30):
    outimage = copy.deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage

def extract_outer(image,Nxc,Nyc, r=30):
    outimage = copy.deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2<=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage

def extract_ring(image, Nxc,Nyc,rin=30,rout=50):
    outimage = copy.deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - rin**2<=0)] = 0
    masked[np.where(x**2 + y**2 - rout**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)

    return outimage

def quad_interp_radius(r_max, dr, val_list):
    v_L = val_list[0]
    v_max = val_list[1]
    v_R = val_list[2]
    rpk = r_max + dr*(v_L - v_R) / (2 * (v_L + v_R - 2*v_max))
    vpk = 8*v_max*(v_L + v_R) - (v_L - v_R)**2 - 16*v_max**2
    vpk /= (8*(v_L + v_R - 2*v_max))
    return (rpk, vpk)

def calc_width(tmpIr,radial,rpeak):
    spline = interpolate.UnivariateSpline(radial, tmpIr-0.5*tmpIr.max(), s=0)
    roots = spline.roots()  # find the roots

    if len(roots) == 0:
        return(radial[0], radial[-1])

    rmin = radial[0]
    rmax = radial[-1]
    for root in np.sort(roots):
        if root < rpeak:
            rmin = root
        else:
            rmax = root
            break

    return (rmin, rmax)

def fit_ring(image,Nr=50,Npa=25,rmin_search = 10,rmax_search = 100,fov_search = 0.1,Nserch =20):
    # rmin_search,rmax_search must be diameter
    image_blur = image.blur_circ(2.0*RADPERUAS,fwhm_pol=0)
    image_mod = image_blur.threshold(cutoff=0.05)
    image_mod = image
    xc,yc = eh.features.rex.findCenter(image_mod, rmin_search=rmin_search, rmax_search=rmax_search,
                         nrays_search=Npa, nrs_search=Nr,
                         fov_search=fov_search, n_search=Nserch)
    return xc,yc

# polarization functions ##############################
def make_polar_imarr(imarr, dx, xc=None, yc=None, rmax=50, Nr=50, Npa=180, kind="linear"):
    '''
    Image array with polar coordinates
    Args:
        imarr (np.ndarray): 1dimensional image array. shape=(ny, nx)
        dx (float): pixel size with the x, y axis
        xc,yc: center of the ring
        rmax (float): maximum radial coordinate for the polar coordinates
        Nr, Npa (int): pixel number of the polar coordinates
        kind (str): kind of interpolation for polar coordinates
    Return:
        radial_imarr (np.ndarray)
    '''
    nx,ny = imarr.shape
    dy=dx
    x= np.arange(nx)*dx/RADPERUAS
    y= np.arange(ny)*dy/RADPERUAS
    #xc, yc =(np.max(x)-np.min(x))/2, (np.max(y)-np.min(y))/2
    if xc==None or yc==None:
    # Compute the image center
        xc,yc = fit_ring(image)

    z = imarr
    f_image = interpolate.interp2d(x,y,z,kind=kind)

    # Create a mesh grid in polar coordinates
    radial_imarr = np.zeros([Nr,Npa])
    pa = np.linspace(0,360,Npa)
    pa_rad = np.deg2rad(pa)
    radius = np.linspace(0,rmax,Nr)
    dr = radius[-1]-radius[-2]

    # interpolation with polar coordinates
    Rmesh, PAradmesh = np.meshgrid(radius, pa_rad)
    x, y = Rmesh*np.sin(PAradmesh) + xc, Rmesh*np.cos(PAradmesh) + yc
    for ir in range(Nr):
        z = [f_image(x[ipa][ir],y[ipa][ir]) for ipa in range(Npa)]
        radial_imarr[ir,:] = z[:]
    radial_imarr = np.fliplr(radial_imarr)

    return radial_imarr,radius, pa

def extract_pol_quantites(im,xc=None, yc=None, blur_size=-1):
    '''
    Calculate polarization quantites with the input image object and blur size in the unit of uas
    Returns:
        net fractional linear polarization, image averaged linear polarization, EVPA (deg), beta2 magnitude and phase (deg), fractional circular polarization
    '''
    Itot, Qtot, Utot = sum(im.imvec), sum(im.qvec), sum(im.uvec)
    if len(im.vvec)==0:
        print("Caution: no stokes V")
        im.vvec = np.zeros_like(im.imvec)
    Vtot = sum(im.vvec)
    # net fractional linear polarization
    mnet=np.sqrt(Qtot*Qtot + Utot*Utot)/Itot

    # image averaged linear polarization
    if blur_size<0:
        mavg = sum(np.sqrt(im.qvec**2 + im.uvec**2))/Itot
    else:
        im_blur = im.blur_circ(blur_size*eh.RADPERUAS, fwhm_pol=blur_size*eh.RADPERUAS)
        mavg = sum(np.sqrt(im_blur.qvec**2 + im_blur.uvec**2))/np.sum(im_blur.imvec)

    # evpa
    evpa =  (180./np.pi)*0.5*np.angle(Qtot+1j*Utot)

    # fractional circular polarization
    vnet = np.abs(Vtot)/Itot

    # beta2
    P = im.qvec+ 1j*im.uvec
    P_radial, radius, pa = make_polar_imarr(P.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc)
    I_radial, dummy, dummy = make_polar_imarr(im.imvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc)
    V_radial, dummy, dummy = make_polar_imarr(im.vvec.reshape(im.xdim, im.xdim), dx=im.psize, xc=xc, yc=yc)
    Pring, Vring, Vring2, Iring = 0, 0, 0, 0
    for ir, ipa in itertools.product(range(len(radius)), range(len(pa))):
        Pring += P_radial[ir, ipa] * np.exp(-2*1j*np.deg2rad(pa[ipa])) * radius[ir]
        Vring2 += V_radial[ir, ipa] * np.exp(-2*1j*np.deg2rad(pa[ipa])) * radius[ir]
        Vring  += V_radial[ir, ipa] * np.exp(-1*1j*np.deg2rad(pa[ipa])) * radius[ir]
        Iring += I_radial[ir, ipa] * radius[ir]
    beta2 = Pring/Iring
    beta2_abs, beta2_angle = np.abs(beta2), np.rad2deg(np.angle(beta2))

    beta2_v = Vring2/Iring
    beta2_v_abs, beta2_v_angle = np.abs(beta2_v), np.rad2deg(np.angle(beta2_v))
    beta_v = Vring/Iring
    beta_v_abs, beta_v_angle = np.abs(beta_v), np.rad2deg(np.angle(beta_v))

    # output dictionary
    outputs = dict(
        time_utc = im.time,
        mnet = mnet,
        mavg = mavg,
        evpa = evpa,
        beta2_abs = beta2_abs,
        beta2_angle = beta2_angle,
        vnet = vnet,
        beta_v_abs = beta_v_abs,
        beta_v_angle = beta_v_angle,
        beta2_v_abs = beta2_v_abs,
        beta2_v_angle = beta2_v_angle
        )
    return outputs

# polarization functions ##############################

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

obs = eh.obsdata.load_uvfits(args.data)
obs.add_scans()
obs = obs.avg_coherent(60.0)

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

paths={}
if args.truthmv!='':
    paths['truth']=args.truthmv
if args.mv!='':
    paths['recon']=args.mv
######################################################################

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

colors = {  
            'truth'     : 'black',
            'recon'     : 'red',
        }

labels = {  
            'truth'    : 'Truth',
            'recon'    : 'Reconstruction',
        }
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32,8), sharex=True)

ax[0,0].set_ylabel('Diameter $({\mu as})$')
ax[0,1].set_ylabel('FWHM $({\mu as})$')
ax[0,2].set_ylabel('Position angle ($^\circ$)')

ax[1,0].set_ylabel('Frac. Central Brightness')
ax[1,1].set_ylabel('Asymmetry')
ax[1,2].set_ylabel('Peak PA ($^\circ$)')

ax[1,0].set_xlabel('Time (UTC)')
ax[1,1].set_xlabel('Time (UTC)')
ax[1,2].set_xlabel('Time (UTC)')


polpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.im_list()[0]
    if len(im.ivec)>0 and len(im.qvec)>0 and len(im.uvec)>0 :
        polpaths[p]=paths[p]

polvpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.im_list()[0]
    if len(im.ivec)>0 and len(im.vvec)>0:
        polvpaths[p]=paths[p]

for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])

    imlist = [mv.get_image(t) for t in times]

    mv_ave = mv.avg_frame()
    # find ring center with the averaged image
    xc,yc = fit_ring(mv_ave)
    ring_outputs = [extract_ring_quantites(im_f,xc=xc,yc=yc) for im_f in tqdm.tqdm(imlist)]
    table = pd.DataFrame(ring_outputs, columns=["time_utc", "D","Derr","W","Werr","PAori","PAerr","papeak","A","Aerr","fc","xc","yc","fwhm_maj","fwhm_min","hole_flux","outer_flux","ring_flux","totalflux","hole_dflux","outer_dflux","ring_dflux"])
    #table_vals[p]=np.round(np.mean(np.array(mnet_tab)),3)

    mc=colors[p]
    alpha = 0.5
    lc=colors[p]
    # Diameter
    ax[0,0].plot(times, table["D"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
    # FWHM
    ax[0,1].plot(times, table["W"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
    # Position angle
    ax[0,2].plot(times, table["PAori"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
    # Frac Central Brightness
    ax[1,0].plot(times, table["fc"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
    # Asymetry
    ax[1,1].plot(times, table["A"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
    # peak position angle
    ax[1,2].plot(times, table["papeak"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)

ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2., 1.3), markerscale=2.5)
plt.savefig(outpath, bbox_inches='tight', dpi=300)


######################################################################
# Stokes QUV
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(32,8), sharex=True)
#"mnet", "mavg", "evpa", "beta2_abs", "beta2_angle", "vnet"
ax[0,0].set_ylabel('$|m|_{net}$')
ax[0,1].set_ylabel(r'$\langle |m|\rangle$')
ax[0,2].set_ylabel('$\chi\ (^\circ)$')

ax[1,0].set_ylabel(r'$\beta_2$')
ax[1,1].set_ylabel(r'$\angle \beta_2\ (^\circ)$')
ax[1,2].set_ylabel('$|v|_{net}$')

ax[1,0].set_xlabel('Time (UTC)')
ax[1,1].set_xlabel('Time (UTC)')
ax[1,2].set_xlabel('Time (UTC)')

#ax[0].set_ylim(0.0,0.1)
#ax[1].set_ylim(0.0,0.15)
#ax[2].set_ylim(0.0,0.05)
#ax[3].set_ylim(0.0,0.1)


pol_count = 0
for p in polpaths.keys():
    mv=eh.movie.load_hdf5(polpaths[p])

    imlist = [mv.get_image(t) for t in times]

    if len(imlist[0].qvec)==len(imlist[0].ivec):
        pol_count += 1
        mv_ave = mv.avg_frame()
        # find ring center with the averaged image
        xc,yc = fit_ring(mv_ave)
        pol_outputs = [extract_pol_quantites(im_f,xc=xc,yc=yc) for im_f in tqdm.tqdm(imlist)]

        # if kine like shifting center, need use the following
        # pol_outputs = [extract_pol_quantites(image_f) for image_f in tqdm.tqdm(ims_rg[0].im_list())]

        table = pd.DataFrame(pol_outputs, columns=["time_utc","mnet","mavg","evpa","beta2_abs","beta2_angle","vnet","beta_v_abs", "beta_v_angle","beta2_v_abs", "beta2_v_angle"])

        mc=colors[p]
        alpha = 0.5
        lc=colors[p]
        # Diameter
        ax[0,0].plot(times, table["mnet"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
        # FWHM
        ax[0,1].plot(times, table["mavg"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
        # Position angle
        ax[0,2].plot(times, table["evpa"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
        # Frac Central Brightness
        ax[1,0].plot(times, table["beta2_abs"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
        # Asymetry
        ax[1,1].plot(times, table["beta2_angle"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
        # peak position angle
        ax[1,2].plot(times, table["vnet"],  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)

if pol_count>0:
    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2., 1.3), markerscale=2.5)
    plt.savefig(outpath[:-4]+'_pol.png', bbox_inches='tight', dpi=300)