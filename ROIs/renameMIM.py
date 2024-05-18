# # Python script to rename ROI folders

# import os
# import glob
# import sys

# # Get ROI folder names
# foldernames = glob.glob(rf"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE\INNOVATE ROIs NT/*.*")

# basenames = [os.path.basename(f) for f in foldernames]



# XXXs = [basename[:3] for basename in basenames]
# YYYs = [basename[4:7] for basename in basenames]

# PatNums = [f'{XXX}_{YYY}' for (XXX,YYY) in zip(XXXs, YYYs)]


# # Rename
# for indx, foldername in enumerate(foldernames):
    
#     print(foldername)
#     print(rf'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE\INNOVATE ROIs NT/{PatNums[indx]}')
#     try:
#         os.rename(foldername, rf'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE\INNOVATE ROIs NT/{PatNums[indx]}')
#     except:
#         None