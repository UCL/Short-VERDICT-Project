# Python script to save edits on mha file as npy
import numpy as np
import SimpleITK as sitk

# Read output folder
OutputFolder = str(open('output_folder.txt', 'r').read())

ROIfolder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\INNOVATE\ROIs"

# Patient number
PatNum = 'INN_291'

# ROI drawer 
ROIdrawer = 'NT'

# ROI name
ROIname = 'L1_b3000_NT'

# Load ROI
ROI = sitk.GetArrayFromImage(sitk.ReadImage(f'{ROIfolder}/{ROIdrawer}/mha/{PatNum}/{ROIname}.mha'))

print(np.sum(ROI))
# save as npy
np.save(f'{ROIfolder}/{ROIdrawer}/numpy/{PatNum}/{ROIname}.npy', ROI)