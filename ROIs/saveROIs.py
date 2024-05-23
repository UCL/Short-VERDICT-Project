# Python script to save ROIs Natasha sent me as npy and mha masks

# Import relevant libraries
import numpy
import glob
import sys
import os

# Import fIC analysis functions
sys.path.insert(0, r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Sensitivity Analysis")
import sensitivity # type: ignore


# ROI folder
ROIfolder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\INNOVATE\ROIs"

# Data folder = 
datafolder = r"C:\Users\adam\OneDrive\Desktop\INNOVATE STUDY COHORT VERDICT IMAGES"

# ROI drawer
ROIdrawer = 'NT'

# ROI Name
ROIName = 'L1_b3000_contour'



# # Find list of patients
# fnames = glob.glob(rf"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE\INNOVATE ROIs NT/*")

# PatNums = [ os.path.basename(fname) for fname in fnames  ]
    
PatNums = ['BAR_054']
# PatNums = [
#     'INN_353',
#     'INN_361',
#     'INN_363',
#     'INN_369',
#     'INN_373',
#     'INN_376',
#     'INN_377',
#     'INN_378',
#     'INN_318',
#     'INN_334',
#     'INN_340',
#     'INN_341',
#     'INN_345',
#     'INN_351',
#     'INN_352',
#     'BAR_020',
#     'INN_310',
#     'INN_321',
#     'INN_331',
#     'INN_335',
#     'INN_339',
#     'INN_342',
#     'INN_239',
#     'INN_244',
#     'INN_247',
#     'INN_249',
#     'INN_303',
#     'INN_322',
#     'INN_180',
#     'INN_182',
#     'INN_299',
#     'INN_306',
#     'INN_329',
#     'INN_364'
# ]
    
# Attempt to save ROI mask for each patient
for PatNum in PatNums:
    
    print(PatNum)
    sensitivity.saveROImask(
        PatNum, 
        ROIdrawer = ROIdrawer,
        ROIName = ROIName,
        ROIfolder = ROIfolder,
        datafolder = datafolder)


    # try:
    #     # Lesion mask
    #     sensitivity.saveROImask(PatNum, ROIName = ROIName)
    #     # print(f'ROI saved for {PatNum}')
        
    # except:
    #     print(f'ROI error for {PatNum}')