# Python script to find list of downloaded patients

import glob
import os
import pandas as pd

INNOVATE_folder = r"D:\UCL PhD Imaging Data\INNOVATE VERDICT"


PatList = [os.path.basename(path) for path in glob.glob(f'{INNOVATE_folder}/*')]

DF = pd.DataFrame(PatList)

DF.to_excel(r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\DownloadedPatNums.xlsx", index = False, header = False)