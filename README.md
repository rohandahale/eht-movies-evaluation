# Evaluation Script for EHT Movies

- Clone this repository
- Use the `environment.yml` to create a new conda env with correct packages
    - `conda env create -n <environment name> -f environment.yml`
    - `conda activate <environment name>`
- Install [ehtplot](https://github.com/liamedeiros/ehtplot)
- Install latest Julia 1.10.2 with [juliaup](https://github.com/JuliaLang/juliaup)
- Make sure the FFMPEG in this conda env works for you. Otherwise install it.
- You just need to modify `driver.py` and run it.

## Input of driver.py

```
# Results Directory
resultsdir='results'

modelname = ...     # Options: 'crescent', 'ring', 'disk', 'edisk', 'double', 'point', 'sgra'
modeltype = ...     # Options: 'ring', 'non-ring'

# Physical CPU cores to be used
cores = 100

# Reconstruction .hdf5 path
pathmov = ...
# Truth .hdf5 path
pathmovt = ....
# Unprocessed Data .uvfits path
data = ...

# Specify whether data was descattered or no scattering was in the data
## scat = 'none': We are dealing with dataset with no scattering added
## scat = 'sct': We are dealing with dataset with scattering added
## scat = 'dcst': There was scattering in original data but it was descattered prior to imaging or during imaging.

scat = ... # Options: sct, dsct, none 


```

## Running the script
`python driver.py`

## Output

Following will be saved in the `resultsdir` defined in the `driver.py`.

- chisq for all Stokes parameters
- Amplitudes at three baselines for all Stokes parameters
- Closure Phases for three trinagles for all Stokes parameters
- For synthetic data, NXCORR for all Stokes parameters
- Stokes I, P, V GIF
- Net and average linear and circular polarization
- REx ring extraction plots
- VIDA IDFE .csv and plots