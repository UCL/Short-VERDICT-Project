# Python script to save editted fIC ROI

import numpy as np
import SimpleITK as sitk
import glob
import os

# Read output folder
OutputFolder = str(open('output_folder.txt', 'r').read())


# Find list of patient numbers
PatNums = [os.path.basename(path) for path in 
           glob.glob(r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Outputs\fIC ROIs\*")]
PatNums = ['BAR_018']
print(PatNums)
for PatNum in PatNums:

    # Load mha ROI
    try:
        ROI = sitk.GetArrayFromImage( sitk.ReadImage(f'{OutputFolder}/fIC ROIs/{PatNum}\L1_b3000_NT.mha'))
        # Save as numpy array
        np.save(f'{OutputFolder}/fIC ROIs/{PatNum}\L1_b3000_NT.npy', ROI)
    except: 
        None
