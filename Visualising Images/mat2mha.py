# Python script to convert .mat image volume files to mha so they can be viewed in ITK SNAP

# Import relevant libraries
import SimpleITK as sitk
from scipy.io import loadmat
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt

# Read output folder
OutputFolder = str(open('output_folder.txt', 'r').read())

# Find list of patient numbers
PatNums = [os.path.basename(path) for path in 
           glob.glob(r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Outputs\fIC ROIs\*")]

PatNums = ["INN_291"]#, "INN_175", "INN_145", "INN_209", "INN_241"]


model = 'New VERDICT'  

modeltype = 'Original VERDICT'

schemename = 'Original Full'

fittingtechnique = 'AMICO'

noisetype = 'Rice'

sigma0train = 0.05

T2train = 10000

VolumeName = 'fIC'

normalise = True

for PatNum in PatNums:
    print(PatNum)
        
    # try:

    if fittingtechnique == 'AMICO':
       volume = loadmat(f'{OutputFolder}/{model} outputs/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{VolumeName}.mat')[VolumeName]
    else:
        # Load .mat file
        volume = loadmat(f'{OutputFolder}/{model} outputs/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/{VolumeName}.mat')[VolumeName]

    # remove infinities
    volume[volume == np.inf] = 0
    # Remove nan
    volume[np.isnan(volume)] = 0
    
    
    if VolumeName == 'T2':
        volume[volume>1000] = 1000
    

    # Change image orientation
    volume = np.moveaxis( volume , -1, 0)  
    
    if normalise:
        volume[volume>1] = 1
        volume[volume<0] = 0
        
    # Save as mha file
    if fittingtechnique == 'AMICO':
        sitk.WriteImage( sitk.GetImageFromArray(volume), f'{OutputFolder}/{model} outputs/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{VolumeName}.mha' )
    else: 
        sitk.WriteImage( sitk.GetImageFromArray(volume), f'{OutputFolder}/{model} outputs/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/{VolumeName}.mha' )
        
    del volume
    
    # except: None




