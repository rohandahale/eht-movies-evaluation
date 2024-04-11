##############################################################################################
# Author: Rohan Dahale, Date: 05 April 2024, Version=v0.9
##############################################################################################

import os

# Dictionary of vida templates available
modelsvida={
        'crescent'  : 'mring_0_1', 
        'ring'      : 'mring_0_0', 
        'disk'      : 'disk_1', 
        'edisk'     : 'stretchdisk_1',
        'double'    : 'gauss_2', 
        'point'     : 'gauss_2',
        'sgra'      : 'stretchmring_1_4'
        }

# Results Directory
resultsdir='results'

modelname = 'crescent' # 'crescent', 'ring', 'disk', 'edisk', 'double', 'point', 'sgra'
modeltype = 'ring'     # 'ring', 'non-ring' (For REx)
template = modelsvida[modelname]

# Physical CPU cores to be used
cores = 100

# Reconstruction .hdf5 path
pathmov = '/mnt/disks/shared/eht/sgra_dynamics_april11/submission_VL_1/ngmem/crescent_3601_LO_thermal+phasegains/crescent_3601_LO_1.hdf5'
# Truth .hdf5 path
pathmovt = '/mnt/disks/shared/eht/sgra_dynamics_april11/submission_VL_1/truth/crescent_3601_LO/crescent_3601_LO.hdf5'
# Unprocessed Data .uvfits path
data = '/mnt/disks/shared/eht/sgra_dynamics_april11/submission_VL_1/data/crescent_3601_LO.uvfits'
scat = 'none'   # Options: sct, dsct, none

##############################################################################################
# Directory of the results
##############################################################################################
if not os.path.exists(f'./{resultsdir}'):
    os.makedirs(f'./{resultsdir}')


##############################################################################################
# Chi-squares, closure triangles, ampltitudes
##############################################################################################
pollist=['I', 'Q', 'U', 'V']
for pol in pollist:
    #########################
    #CHISQ
    #########################
    outpath=f'./{resultsdir}/chisq_{pol}_{modelname}.png'
    if not os.path.exists(outpath):
        os.system(f'python ./src/chisq.py -d {data} --mv {pathmov} -o {outpath} --pol {pol} --scat {scat}')
    
    #########################
    # CPHASE
    #########################
    outpath_tri=f'./{resultsdir}/triangle_{pol}_{modelname}.png'
    if not os.path.exists(outpath_tri):
        os.system(f'python ./src/triangles.py -d {data} --mv {pathmov} -o {outpath_tri} --pol {pol} --scat {scat}')
    
    #########################
    # AMP
    #########################
    outpath_amp=f'./{resultsdir}/amplitude_{pol}_{modelname}.png'
    if not os.path.exists(outpath_amp):
        os.system(f'python ./src/amplitudes.py -d {data} --mv {pathmov} -o {outpath_amp} --pol {pol} --scat {scat}')
        
##############################################################################################
# NXCORR
##############################################################################################

if modelname!='sgra':
    outpath =f'./{resultsdir}/nxcorr_{modelname}.png'
    if not os.path.exists(outpath):
        os.system(f'python ./src/nxcorr.py --data {data} --truthmv {pathmovt} --mv {pathmov} -o {outpath} --scat {scat}')

##############################################################################################      
# Stokes I GIF
##############################################################################################
outpath =f'./{resultsdir}/gif_{modelname}.gif'
if not os.path.exists(outpath):
    if modelname!='sgra':
        os.system(f'python ./src/gif.py --data {data} --truthmv {pathmovt} --mv {pathmov} -o {outpath} --scat {scat}')
    else:
        os.system(f'python ./src/gif.py --data {data} --mv {pathmov} -o {outpath} --scat {scat}')


# Stokes P GIF 
outpath =f'./{resultsdir}/gif_lp_{modelname}.gif'
if not os.path.exists(outpath):
    if modelname!='sgra':
        os.system(f'python ./src/gif_lp.py --data {data} --truthmv {pathmovt} --mv {pathmov} -o {outpath} --scat {scat}')
    else:
        os.system(f'python ./src/gif_lp.py --data {data} --mv {pathmov} -o {outpath} --scat {scat}')

##############################################################################################
# Stokes V GIF
##############################################################################################
outpath =f'./{resultsdir}/gif_cp_{modelname}.gif'
if not os.path.exists(outpath):
    if modelname!='sgra':
        os.system(f'python ./src/gif_cp.py --data {data}  --truthmv {pathmovt} --mv {pathmov} -o {outpath} --scat {scat}')
    else:
        os.system(f'python ./src/gif_cp.py --data {data} --mv {pathmov} -o {outpath} --scat {scat}')

##############################################################################################
# Pol net, avg 
##############################################################################################
outpath =f'./{resultsdir}/pol_{modelname}.png'
if not os.path.exists(outpath):
    if modelname!='sgra':
        os.system(f'python ./src/pol.py --data {data} --truthmv {pathmovt} --mv {pathmov} -o {outpath} --scat {scat}')
    else:
        os.system(f'python ./src/pol.py --data {data} --mv {pathmov} -o {outpath} --scat {scat}')

##############################################################################################    
# REx ring characterization
##############################################################################################
if modeltype =='ring':
    outpath =f'./{resultsdir}/rex_{modelname}.png'
    if not os.path.exists(outpath) and not os.path.exists(outpath[:-4]+'.png'):
        if modelname!='sgra':
            os.system(f'python ./src/rex.py --data {data} --truthmv {pathmovt} --mv {pathmov} -o {outpath}')
        else:
            os.system(f'python ./src/rex.py --data {data} --mv {pathmov} -o {outpath}')
        
##############################################################################################        
# VIDA
##############################################################################################
input  = pathmov
output = f'./{resultsdir}/{modelname}_vida.csv'
if not os.path.exists(output):
    os.system(f'julia -p {cores} ./src/movie_extractor_parallel.jl --input {input} --output {output} --template {template} --stride {cores}')
input_t  = pathmovt
output_t = f'./{resultsdir}/{modelname}_truth_vida.csv'
if not os.path.exists(output_t):
    os.system(f'julia -p {cores} ./src/movie_extractor_parallel.jl --input {input_t} --output {output_t} --template {template} --stride {cores}')
                        
truthcsv  = output_t
mvcsv     = output

outpath =f'./{resultsdir}/vida_{modelname}.png'
if not os.path.exists(outpath):
    if modelname!='sgra':    
        os.system(f'python ./src/vida.py --model {modelname} --truthcsv {truthcsv} --mvcsv {mvcsv} -o {outpath}')
    else:
        os.system(f'python ./src/vida.py --model {modelname} --truthcsv {truthcsv} --mvcsv {mvcsv} -o {outpath}')


##############################################################################################
# Interpolated Movie, Averaged Movie, VIDA Ring, Cylinder
##############################################################################################
if not os.path.exists(f'./{resultsdir}/patternspeed'):
    os.makedirs(f'./{resultsdir}/patternspeed')
if not os.path.exists(f'./{resultsdir}/patternspeed_truth'):
    os.makedirs(f'./{resultsdir}/patternspeed_truth')
    
# Interpolated Movies
input=pathmov
output=f'./{resultsdir}/patternspeed/{os.path.basename(pathmov)}'
os.system(f'python ./src/hdf5_standardize.py -i {input} -o {output}')
input=pathmovt
output=f'./{resultsdir}/patternspeed_truth/{os.path.basename(pathmovt)}'
os.system(f'python ./src/hdf5_standardize.py -i {input} -o {output}')
        
#Average Movies
input=f'./{resultsdir}/patternspeed/{os.path.basename(pathmov)}'
fits=os.path.basename(pathmov)[:-5]+'.fits'
output=f'./{resultsdir}/patternspeed/{fits}'
os.system(f'python ./src/avg_frame.py -i {input} -o {output}')
input=f'./{resultsdir}/patternspeed_truth/{os.path.basename(pathmovt)}'
fits=os.path.basename(pathmovt)[:-5]+'.fits'
output=f'./{resultsdir}/patternspeed_truth/{fits}'
os.system(f'python ./src/avg_frame.py -i {input} -o {output}')

# VIDA Ring
fits=os.path.basename(pathmov)[:-5]+'.fits'
path=f'./{resultsdir}/patternspeed/{fits}'
outpath = path[:-5]+'.csv'
if not os.path.exists(outpath):    
    os.system(f'julia ./src/ring_extractor.jl --in {path} --out {outpath}')
    print(f'{os.path.basename(outpath)} created!')
    
fits=os.path.basename(pathmovt)[:-5]+'.fits'
path=f'./{resultsdir}/patternspeed_truth/{fits}'
outpath = path[:-5]+'.csv'
if not os.path.exists(outpath):    
    os.system(f'julia ./src/ring_extractor.jl --in {path} --out {outpath}')
    print(f'{os.path.basename(outpath)} created!')

# Cylinder
ipathmov=f'./{resultsdir}/patternspeed/{os.path.basename(pathmov)}'
ipathmovt=f'./{resultsdir}/patternspeed_truth/{os.path.basename(pathmovt)}'
paths=[pathmovt, pathmov]
  
ringpath = ipathmov[:-5]+'.csv'
outpath  = ipathmov[:-5]
os.system(f'python ./src/cylinder.py {ipathmov} {ringpath} {outpath}')

ringpath = ipathmovt[:-5]+'.csv'
outpath  = ipathmovt[:-5]
os.system(f'python ./src/cylinder.py {ipathmovt} {ringpath} {outpath}')