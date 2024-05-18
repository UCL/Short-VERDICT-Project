# Python script to run fIC analysis functions

# Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import pickle
from scipy.io import savemat
import pandas as pd

# Import fIC analysis functions
sys.path.insert(0,r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Sensitivity Analysis")
import sensitivity #type: ignore

# Read output folder
OutputFolder = str(open('output_folder.txt', 'r').read())

# OutputFolder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs"

# ===================================================


# Define ROI drawer
ROIdrawer = 'NT'

# Define ROI averaging type
avg_type = 'median'

# Define Model numbers to analyse
ModelNames= ['Original VERDICT, All scans',
             'Original VERDICT, ex. 90, 3000',
             'No VASC VERDICT, ex. 90, 3000',
             'RDI, ex. 90, 3000',
             'RDI, ex. 90, 500, 3000'
             ]



plt.figure()

# Loop over model numbers
for ModelName in ModelNames:

    # ROI Name
    if ROIdrawer == 'NT':
        ROIName = 'L1_b3000_NT'
    else:
        ROIName = 'L1'


    # Find list of patients
    fnames = glob.glob(f'{OutputFolder}\VERDICT outputs\*')#/Model {ModelNum}/fIC.mat')

    PatNums = [ os.path.basename(fname) for fname in fnames  ]

    # Create list for 'good PatNums' which have a well defined ROI from Natasha
    goodPatNums = []

    # For each patient, save lesion ROI masks and extract fICs
    for indx, PatNum in enumerate(PatNums):
        
        # print(f'== {PatNum}')
        
        # # Excluding ROIs
        if PatNum in ['INN_291', 'INN_167', 'INN_155']:
            continue
        
        # If ROI mask exists, try and extract ROI 
        try:
            
            '''Using Natasha's RTstruct ROIs'''
            # # Extract fICs
            if ROIdrawer == 'NT':
                sensitivity.extractROIfICs(PatNum, 
                                            ROIName = ROIName, 
                                            ModelName = ModelName,
                                            parameter = 'fIC',
                                            MaskType = 'numpy',
                                            VERDICT_output_path = rf"{OutputFolder}\VERDICT outputs",
                                            ROI_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs\ROIs",
                                            # ROI_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Outputs\fIC ROIs",
                                            output_path = rf"{OutputFolder}\fIC results"
                                            )
            
            
            # '''Using my ROIs'''
            # if ROIdrawer == 'AP':
            #     fIC_analysis.extractROIfICs(
            #         PatNum, 
            #         ROIName = ROIName,
            #         ModelNum = ModelNum,
            #         MaskType = 'Analyze',
            #         ROI_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE\INNOVATE my ROIs"
            #         )
                
            # Append to goodPatNums list
            goodPatNums.append(PatNum)
            
        # If no ROI RTstruct file, or error, remove patient number from list
        except:
            print(f'No ROI for {PatNum}')

    
    PatNums = goodPatNums


    # == Extract average and variance of fICs in ROIs

    # Calculate median fICs  
    sensitivity.avgROIfICs(
        PatNums = PatNums, 
        ROIName = ROIName, 
        ModelName = ModelName, 
        avg_type = avg_type,
        results_path = rf"{OutputFolder}\fIC results"
        )


    # # fIC variance across ROI
    # fIC_analysis.ROIvariance(PatNums = PatNums, ROIName = ROIName, ModelNum = ModelNum)



    # Apply ROC analysis
    ResultsDF, fpr, tpr, thresholds, roc_auc = sensitivity.fIC_ROC( 
                                                                    ModelName = ModelName, 
                                                                    avg_type = avg_type,
                                                                    results_path = rf"{OutputFolder}\fIC results"
                                                                    )


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(ResultsDF)
        
    # == Save results

    # Make folder
    try:
        os.makedirs(f'{OutputFolder}/ROC results/{ROIdrawer}/{ModelName}/{avg_type}')
    except: 
        None
        
    # Save results dataframe as pickle
    with open(f'{OutputFolder}/ROC results/{ROIdrawer}/{ModelName}/{avg_type}/Results_DF.pickle', 'wb') as handle:
        pickle.dump(ResultsDF, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Save results data frame as matlab structure
    savemat(f'{OutputFolder}/ROC results/{ROIdrawer}/{ModelName}/{avg_type}/ResultsDF.mat', {'ResultsDF': ResultsDF.to_dict(orient = 'list')})
        
    # Save ROC results
    np.save(f'{OutputFolder}/ROC results/{ROIdrawer}/{ModelName}/{avg_type}/fpr.npy', fpr)
    np.save(f'{OutputFolder}/ROC results/{ROIdrawer}/{ModelName}/{avg_type}/tpr.npy', tpr)
    np.save(f'{OutputFolder}/ROC results/{ROIdrawer}/{ModelName}/{avg_type}/thresholds.npy', thresholds)
    np.save(f'{OutputFolder}/ROC results/{ROIdrawer}/{ModelName}/{avg_type}/roc_auc.npy', roc_auc)
    
    print(roc_auc)

    Youden_Indices = tpr-fpr

    print(f'ROC AUC: {roc_auc}')
    print(f'Thresholds: {thresholds}')
    print(f' Best threshold: {thresholds[Youden_Indices == np.max(Youden_Indices)]}')
    
    plt.plot(fpr,tpr, label = f'{ModelName}, AUC: {str(round(roc_auc,3))}')
    
    
plt.grid('on')
plt.legend()
plt.show()


