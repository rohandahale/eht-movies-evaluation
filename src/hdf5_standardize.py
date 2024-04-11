#####################################################################
# Author: Rohan Dahale, Marianna Foschi, Date: 25 Mar 2024
######################################################################

# Import libraries
import numpy as np
import ehtim as eh
import argparse
import glob
import os

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str, help='path of input original hdf5 file')
    p.add_argument('-o', '--output', type=str, help='path of output standard hdf5 file')
    return p

######################################################################
# List of parsed arguments
args = create_parser().parse_args()
npix   = 128
fov    = 200 * eh.RADPERUAS
ntimes = 100
tstart = 10.91 
tstop  = 13.90 #14.04 

# tstart and tstop from obsfile of an example synthetic data
#>>> obs.tstop
#14.041666030883789
#>>> obs.tstart
#10.891666889190674

movie = eh.movie.load_hdf5(args.input)
times = np.linspace(tstart, tstop, ntimes)
frame_list = [movie.get_image(t).regrid_image(fov, npix) for t in times]
new_movie = eh.movie.merge_im_list(frame_list)
new_movie.reset_interp(bounds_error=False)
new_movie.save_hdf5(args.output)
######################################################################
