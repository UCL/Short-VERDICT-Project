# Python script to save ROIs Natasha sent me as npy and mha masks

# Import relevant libraries
import numpy
import glob
import sys
import os

# Import fIC analysis functions
sys.path.insert(0, r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Sensitivity Analysis")
import sensitivity # type: ignore

# ROI drawer
ROIdrawer = 'NT'

# ROI Name
ROIName = 'L1_b3000_NT'

# # Find list of patients
# fnames = glob.glob(rf"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE\INNOVATE ROIs NT/*")

# PatNums = [ os.path.basename(fname) for fname in fnames  ]
    

PatNums = [
    # "INN_110",
#            "INN_112",
#            "INN_114",
#            "INN_128",
#            "INN_129",
#            "INN_134",
           "INN_146",
           "INN_190",
           "INN_226", 
           "INN_238",
           "INN_117"]
    
# Attempt to save ROI mask for each patient
for PatNum in PatNums:
    
    print(PatNum)
    sensitivity.saveROImask(PatNum, ROIdrawer = ROIdrawer,ROIName = ROIName)


    # try:
    #     # Lesion mask
    #     sensitivity.saveROImask(PatNum, ROIName = ROIName)
    #     # print(f'ROI saved for {PatNum}')
        
    # except:
    #     print(f'ROI error for {PatNum}')