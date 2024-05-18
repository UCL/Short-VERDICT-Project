# Python script to contain functions to analyse fIC values with lesion ROI

# Import relevnat libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import sys
import glob
import os
from scipy.io import loadmat, savemat
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Perceptron, LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import SimpleITK as sitk
# from skimage.metrics import mean_squared_error
import scipy
# import skimage

# Import DICOM from imgtools
sys.path.insert(0, r"C:\Users\adam\OneDrive - University College London\UCL PhD\Image-Processing\DICOM\Python")
import DICOM # type: ignore


# Function for saving ROI masks
def saveROImask(
                PatNum, 
                ROIdrawer,
                ROIName, 
                datafolder = r"D:\UCL PhD Imaging Data\INNOVATE VERDICT", 
                ROIfolder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\ROIs",
):
    '''
    Function to save mask of ROI in RTStruct file.
    
    ROIs drawn on b3000 VERDICT scan as this is used as the registration target
    
    '''
    
    try:
        # Define path to image DICOMs
        Img_DICOM_path = rf"{datafolder}\{PatNum}\scans" 
        
        # Find filename for b3000 image
        b3000_DICOM_fnames = glob.glob(rf"{Img_DICOM_path}/*b3000_80/DICOM/*.dcm")

        # Test to throw error if empty list
        test_valid = b3000_DICOM_fnames[0]
 
    except:
        
        # Define path to image DICOMs
        Img_DICOM_path = rf"{datafolder}\{PatNum}" 

        # Find filename for b3000 image
        b3000_DICOM_fnames = glob.glob(f'{Img_DICOM_path}\*b3000_80\DICOM\*.dcm')

        print('2')
        print(b3000_DICOM_fnames)
    # Test if DICOM MF or SF
    if len(b3000_DICOM_fnames) == 1:
        # MF
        multiframe = True
        b3000_DICOM_fname = b3000_DICOM_fnames[0]
        b3000dcm = DICOM.MRIDICOM(b3000_DICOM_fname)
        
    elif len(b3000_DICOM_fnames) > 1:
        # SF
        multiframe = False
        b3000dcm = DICOM.MRIDICOM(DICOM_fnames = b3000_DICOM_fnames, multiframe = multiframe)
        
    else:
        print(f'No b3000 file for patient {PatNum}')
        sys.exit()
        
        
    # Create dcm object 
    b3000_ImageArray = b3000dcm.constructImageArray()
    b3000_bVals = b3000dcm.DICOM_df['Diffusion B Value'].values
    b0 = b3000_ImageArray[b3000_bVals == 0]
    
    # Define path to ROI DICOM
    RTStruct_path = f'{ROIfolder}/{ROIdrawer}/DICOM/{PatNum}'

    # Find RTstruct filename
    RTStruct_fname = glob.glob(f'{RTStruct_path}/*')[0]


    # === Create ROI mask

    # Instantiate contours object
    contours = DICOM.contours(RTStruct_fname)

    # Define lesion structure number (hardcoded here but should be found automatically in future)
    LesionStructNum = contours.Struct_Name_Num_dict[ROIName]
 

    LesionMask = contours.create_mask(Struct_Num = LesionStructNum, DICOM_dcm = b3000dcm)


    # Remove duplicate spatial slices
    LesionMask = LesionMask[b3000_bVals == 0]
    
    
    
    '''New code: accounting for z flips of fIC map'''
    
    # == First, load in b0 data from Matlab output
    
    # # Load .mat file
    # matb0 = loadmat(f'VERDICT outputs/{PatNum}/Model 1/b0from3000.mat')['b0from3000']
    
    # # Permute matlab b0 volume
    # matb0 = np.moveaxis( matb0 , -1, 0)  
    

    # # == Calculate MSE for different relative orientations
    # MSE0 = mean_squared_error(matb0, b0)
    # MSE1 = mean_squared_error(matb0, b0[::-1, : , :])

    # # If MSE0 > MSE1, python b3000 image and mask need to be flipped to match fIC
    # if MSE0 > MSE1:
    #     print('oof')
    #     LesionMask = LesionMask[::-1,:,:]
    #     b0 = b0[::-1,:,:]
    # else:
    #     None
        
    
    # == Save lesion mask as numpy array
    try:
        os.makedirs(f'{ROIfolder}/{ROIdrawer}/numpy/{PatNum}')
    except:
        None
      
    # Save as npy  
    np.save(f'{ROIfolder}/{ROIdrawer}/numpy/{PatNum}/{ROIName}.npy', LesionMask)
    
    # == Save lesion mask as mha
    
    try:
        os.makedirs(f'{ROIfolder}/{ROIdrawer}/mha/{PatNum}')
    except:
        None
        
    sitk.WriteImage( sitk.GetImageFromArray(LesionMask), f'{ROIfolder}/{ROIdrawer}/mha/{PatNum}/{ROIName}.mha' )
    
    # == Save b=0 from b3000 image
    
    # Save as npy  
    np.save(f'{ROIfolder}/{ROIdrawer}/numpy/{PatNum}/b0from3000.npy', b0)
    
    # Save as mha
    sitk.WriteImage( sitk.GetImageFromArray(b0), f'{ROIfolder}/{ROIdrawer}/mha/{PatNum}/b0from3000.mha' )     
 

# Function foimask(
#     PatNum,
#     Nifti_INNOVATE_ROIs_path = r"D:\UCL PhD Imaging Data\Nifti INNOVATE ROIs",
#     output_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\PhD_Year_1\ISMRM Submission\ROIs"):
    
#     # Read Nifti image
#     mask = sitk.GetArrayFromImage(
#         sitk.ReadImage(f'{Nifti_INNOVATE_ROIs_path}/{PatNum}.nii.gz')
#     )
    
#     print(mask.shape)
#     plt.figure()
#     plt.imshow(mask[4])
#     plt.show()


         
# Function for extracting fIC values from ROI
def extractROIvalues(
    PatNum,
    ROIdrawer,
    ROIName,
    modeltype,
    schemename,
    fittingtechnique,
    parameters = 'fIC',  # NEED TO ADAPT THIS TO TKAE MULITPLE PARAMETERS
    MaskType = 'numpy',
    noisetype = 'Rice',
    sigma0train = 0.05,
    T2train = 10000,
    VERDICT_output_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs\VERDICT outputs",
    ROI_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs\ROIs",
    output_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs\fIC results"
):
    
    '''
    Function to extract parameter values from within an ROI
    
    '''    
    
    # if fittingtechnique == 'MLP':
    #     PatNumSig = f'{PatNum}/sigma0 = {sigma0train}'            
    # else:
    #     PatNumSig = PatNum
        
        
    # For multiple parameters
    if type(parameters) != list:       
        parameters = [parameters]
    

        
    for parameter in parameters:
    

        # Load fIC array
        if fittingtechnique == 'AMICO':
            paramMap = loadmat(f'{VERDICT_output_path}/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}//{parameter}.mat')[parameter]
        else:
            paramMap = loadmat(f'{VERDICT_output_path}/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/{parameter}.mat')[parameter]
        
        # remove infinities
        paramMap[paramMap == np.inf] = 0
        
        # Remove nan
        paramMap[np.isnan(paramMap)] = 0
        
        # Permute array axes (account for MATLAB Python differences)
        paramMap = np.moveaxis( paramMap , -1, 0)   
        

        if MaskType == 'numpy':
            # Load ROI mask
            ROIMask = np.load(f'{ROI_path}/{MaskType}/{PatNum}/{ROIName}.npy')
            
        elif MaskType == 'Analyze':
            
            ROIMask = sitk.GetArrayFromImage(sitk.ReadImage(f'{ROI_path}/{MaskType}/{PatNum}/{ROIName}.img'))
            
        else:
            print('Incorrect mask type')
        
        
        # Make Bool
        ROIMask = (ROIMask != 0)
        
        # Extract fIC from ROI
        ROIvalues = paramMap[ROIMask]
        
        # Save as numpy array
        try:
            if fittingtechnique == 'AMICO':
                folder = f'{output_path}/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}//{ROIdrawer}/{ROIName}'
                os.makedirs(folder)               
            else:
                folder = f'{output_path}/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/{ROIdrawer}/{ROIName}'
                os.makedirs(folder)
        except:
            None
            
        # Save as npy
        np.save(f'{folder}/{parameter}s.npy', ROIvalues)
        
    # # Save fIC image as mha
    # sitk.WriteImage(sitk.GetImageFromArray(fIC), f'{VERDICT_output_path}/{PatNum}/{ModelName}/{parameter}.mha')
    
# Function to read biopsy results     
def readBiopsyResults(
    biopsy_data_xlsx = r'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\INNOVATE patient groups 2.0.xlsx',
    results_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Outputs\Biopsy Results"
):
    
    '''
    Function to read biopsy data excel sheet and save results as binary dataframe
    
    '''
    
    BiopsyDF = pd.read_excel(biopsy_data_xlsx)
    
    # Clinically significant patients
    csPats = list( (BiopsyDF['Clinically significant (Gleason >=7)'].values)[1:] )
    
    
    # Ones (binary 1 for cs)
    ones = list(np.ones(len(csPats)))
    
    # Non-clincially significant patients 
    ncsPats = list( (BiopsyDF['Clinically insignificant (Gleason <7)'].values)[1:] )
    
    # Zeros (binary 0 for ncs)
    zeros = list(np.zeros(len(ncsPats)))
    
    
    # Make dataframe
    Patients = csPats + ncsPats
    Results = ones + zeros
    
    BiopsyResultsDF = pd.DataFrame({'Patient_ID': Patients, 'Biopsy_Result': Results})
    
    # Save as matlab structure
    savemat(f'{results_path}/BiopsyResultsDF.mat', {'BiopsyResults': BiopsyResultsDF.to_dict(orient = 'list')})
 
    
def trainPerceptron(
    PatNums,
    parameters,
    datafolder,
    ROIdrawer,
    ROIName, 
    modeltype ,
    schemename,
    fittingtechnique,
    sigma0train,
    parameterscoretype = 'median'
):
    
    '''
    Train perceptron classifier
    
    Need to think about where to save trained perceptron
    
    Maybe in ROIscores/modeltype/schemename/fittingtechnique/ROIdrawer/ROIname/percpetrons/T2_fIC/clf.sav
    
    ===========
    
    Training data
    
    X: array of parameter vectors (e.g. median lesion fICs and T2s)
    Y: vector lof labels (1 for positive biopsy, 0 for negative)
    
    '''
    
    # === Configure X array
    X = np.zeros((len(PatNums), 2))
    
    for patIndx, PatNum in enumerate(PatNums):
        
        for paramIndx, parameter in enumerate(parameters):
        

            
            if fittingtechnique == 'MLP':
                PatNumSig = f'{PatNum}/sigma0 = {sigma0train}'            
            else:
                PatNumSig = PatNum
        
            # Load ROI values
            param_values = np.load(
                f'{datafolder}/ROI Parameter Values/{modeltype}/{schemename}/{fittingtechnique}/{PatNumSig}/{ROIdrawer}/{ROIName}/{parameter}s.npy'
            ) 
            
            # Find median values
            param_median = np.median(param_values)
            
            # Append to X
            X[patIndx, paramIndx] = param_median
     

    # Scale X
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1)) 
    scaler.fit(X)
    X = scaler.transform(X)
    
    # == Configure labels
    
    # Load biopsy results
    BiopsyResultsDF = loadmat(f'{datafolder}/Biopsy Results/BiopsyResultsDF.mat')
    BiopsyResults = BiopsyResultsDF['BiopsyResults']
    
    BR_PatNums = BiopsyResults[0][0][0]
    BR_results = BiopsyResults[0][0][1][0]   

    Y = np.zeros(len(PatNums))
    
    for patIndx, PatNum in enumerate(PatNums):
        
        BR = BR_results[BR_PatNums == PatNum]
        Y[patIndx] = BR
        
    
    # == Train linear classifier
    clf = LogisticRegression()
    clf.fit(X, Y)

    # == Save linear classifier
    
    params_string = ''
    for parameter in parameters:
        params_string += f'{parameter}_'
    
    f'/Perceptron Classifiers/{params_string}/'
    
    try:
        if fittingtechnique == 'MLP':
            folder = f'{datafolder}/Perceptron Classifiers/{params_string}/{modeltype}/{schemename}/{fittingtechnique}/sigma0train = {sigma0train}/{ROIdrawer}/{ROIName}'
        else:
            folder = f'{datafolder}/Perceptron Classifiers/{params_string}/{modeltype}/{schemename}/{fittingtechnique}/{ROIdrawer}/{ROIName}'
            
        os.makedirs(folder)
    except:
        None
        
    with open(f'{folder}/clf.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{folder}/scaler.pickle', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
       
       
            
# Function to calcuate score from ROI
def ROIscore(
    PatNums, 
    modeltype,
    schemename,
    fittingtechnique,
    noisetype,
    sigma0train,
    T2train,
    ROIdrawer,
    ROIName,
    parameters = 'fIC',
    scoretype = 'median',
    datafolder = 'no folder',
    data_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Outputs\ROI Parameter Values",
    results_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs\fIC results" 
):
    
    # if fittingtechnique == 'MLP':
    #     PatNumSigs = []
    #     for PatNum in PatNums:
    #         PatNumSigs.append(f'{PatNum}/sigma0 = {sigma0train}' )    
            
    #     # Extract list of filenames
    #     fnames = glob.glob(f'{data_path}/{modeltype}/{schemename}/{fittingtechnique}/*/sigma0 = {sigma0train}/{ROIdrawer}/{ROIName}/{parameters}s.npy')
             
    # else:
    #     PatNumSigs = PatNums
    #     # Extract list of filenames
    #     fnames = glob.glob(f'{data_path}/{modeltype}/{schemename}/{fittingtechnique}/*/{ROIdrawer}/{ROIName}/{parameters}s.npy')
             
    
    print('yes')
    
    if type(parameters) == str:
        parameters = [parameters]
    
    # For each patient, extract parameter ROI values and calcuate score
    scores = []
    
    
    for PatNum in PatNums:
    #     print(PatNum)    
    #     if fittingtechnique == 'MLP':
    #         PatNumSig = f'{PatNum}/sigma0 = {sigma0train}' 
    #     else:
    #         PatNumSig = PatNum
         
         
        
        if fittingtechnique == 'AMICO':
            folder = f'{data_path}/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{ROIdrawer}/{ROIName}'
        else:
            folder = f'{data_path}/{modeltype}/{schemename}/{fittingtechnique}/{PatNum}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/{ROIdrawer}/{ROIName}'
        
        # Extract parameter values
        ROIvalues = [np.load(f'{folder}/{parameter}s.npy') for parameter in parameters]
        
        
        # Calculate score (ADD OPTIONS HERE)
        if scoretype == 'median':
            score = np.median(ROIvalues[0])
            
            
        # elif scoretype == 'perceptron':
            
        #     params_string = ''
        #     for parameter in parameters:
        #         params_string += f'{parameter}_'   
                      
        #     if fittingtechnique == 'MLP':
        #         folder = f'{datafolder}/Perceptron Classifiers/{params_string}/{modeltype}/{schemename}/{fittingtechnique}/sigma0train = {sigma0train}/{ROIdrawer}/{ROIName}'
        #     else:
        #         folder = f'{datafolder}/Perceptron Classifiers/{params_string}/{modeltype}/{schemename}/{fittingtechnique}/{ROIdrawer}/{ROIName}'
            
        #     with open(f'{folder}/clf.pickle', 'rb') as handle:
        #         clf = pickle.load(handle)             

        #     with open(f'{folder}/scaler.pickle', 'rb') as handle:
        #         scaler = pickle.load(handle)   
                
                
        #     X = np.asarray([[np.median(vals) for vals in ROIvalues]])
        #     Xsc = scaler.transform(X)
        #     score = clf.predict(Xsc)[0]
            
                               
        # Append to scores
        scores.append(score)
        # print(scores)
        

        
    # Create dataframe
    scoresDF = pd.DataFrame({'Patient_ID': PatNums, 'Score': scores})
       

    # if scoretype == 'perceptron':    
    #     scoretype = f'perceptron/{params_string}'


    
    # Create directory
    try:
        if fittingtechnique == 'AMICO':
            path = f'{results_path}/{modeltype}/{schemename}/{fittingtechnique}/{ROIdrawer}/{ROIName}/{scoretype}'
            os.makedirs(path)
        else:
            path = f'{results_path}/{modeltype}/{schemename}/{fittingtechnique}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/{ROIdrawer}/{ROIName}/{scoretype}'
            os.makedirs(path)
    except:
        None
        
    # Save dataframe as pickle
    with open(f'{path}/scoresDF.pickle', 'wb') as handle:
        pickle.dump(scoresDF, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
    # Save as matlab structure
    savemat(f'{path}/scoresDF.mat', {'scores': scoresDF.to_dict(orient = 'list')})
    
    # # # Save dataframe as txt
    # # with open(f'{results_path}/Average fIC Dataframes/Model {ModelNum}/average_fIC_df.txt', 'w') as f:

        
        
# Function to compute variance in fIC across ROI
def ROIvariance(
    PatNums,
    ROIName,
    ModelName,
    results_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs\fIC results"
):
    
    variances = []
    thesePatNums = []
    
    # Iterate over patient numbers
    for PatNum in PatNums:
        
        try:    
            # Load ROI fICs
            fICs = np.load(f'fIC results/{PatNum}/{ROIName}/{ModelName}.npy') 

            # Variance
            var = np.var(fICs)
            variances.append(var)
            
            # If success, append PatNum
            thesePatNums.append(PatNum)
            
        except:
            continue

    # Make array
    variances = np.asarray(variances)
    thesePatNums = np.asarray(thesePatNums)
    
    # Make dataframe
    fICvarDF = pd.DataFrame({'Patient_ID': thesePatNums, 'var_fIC': variances})
    
    # Create directory
    try:
        os.makedirs(f'{results_path}/variance fIC Dataframes/{ModelName}')
    except:
        None
        
    # Save dataframe as pickle
    with open(f'{results_path}/variance fIC Dataframes/{ModelName}/variance_fIC_df.pickle', 'wb') as handle:
        pickle.dump(fICvarDF, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
       
def fIC_ROC(
    ModelName,
    avg_type,
    results_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\ISMRM Submission\Outputs\fIC results"
    ):
    
    '''
    Python function to generate ROC curve for lesion classification from 
    a specified model type
    
    General methods:
    
    1. Read in average fIC and biopsy results dataframes
    2. Create corresponding arrays of average fIC and biopsy outcomes (significant or insignificant)
    3. Use sklearn to generate ROC curve
    
    '''
    
    # Read in average fIC dataframe
    with open(f'{results_path}/{avg_type} fIC Dataframes/{ModelName}/{avg_type}_fIC_df.pickle', 'rb') as handle:
        fIC_DF = pickle.load(handle)
        
    # Read in biopsy results dataframe
    BiopsyResults_DF = readBiopsyResults()

    # Construct list of common patients (in Biopsy DF and fIC DF)
    fIC_PatList = fIC_DF['Patient_ID'].values
    Biopsy_PatList = BiopsyResults_DF['Patient_ID'].values

    PatList = fIC_PatList[ np.array([Pat in Biopsy_PatList for Pat in fIC_PatList]) ]
    
    # Construct arrays for biopsy results and fIC
    BiopsyResults = []
    fICs = []
    
    for PatNum in PatList:
        # Extract fIC
        fIC_Bools = (fIC_DF['Patient_ID'].values == PatNum)
        fICs.append(fIC_DF['Avg_fIC'].values[fIC_Bools][0])
        # Extract biopsy result
        Biopsy_Bools = (BiopsyResults_DF['Patient_ID'].values == PatNum)
        BiopsyResults.append(BiopsyResults_DF['Biopsy_Result'].values[Biopsy_Bools][0])
        
    # Make arrays
    fICs = np.asarray(fICs)    
    BiopsyResults = np.asarray(BiopsyResults)
    
    print(np.sum(BiopsyResults))
    
    # Make dataframe
    ResultsDF = pd.DataFrame({'Patient_ID': PatList, 'Biopsy_Result': BiopsyResults, 'Average_lesion_fIC': fICs })
        
    
    # Create ROC curve
    fpr, tpr, thresholds = roc_curve(y_true = BiopsyResults, y_score = fICs)
    
    # Calculate roc_auc_score
    roc_auc = roc_auc_score(y_true = BiopsyResults, y_score = fICs)
    
    return ResultsDF, fpr, tpr, thresholds, roc_auc


# Function to construct confusion table
def confusionTable(ROIdrawer, ModelNum, avg_type):
    
    # Load results dataframe
    with open(rf'ROC results\{ROIdrawer}\Model {ModelNum}\{avg_type}\Results_DF.pickle', 'rb') as handle:
        ResultDF= pickle.load(handle)

    # Load tpr
    tpr = np.load(rf'ROC results\{ROIdrawer}\Model {ModelNum}\{avg_type}\tpr.npy')
    fpr = np.load(rf'ROC results\{ROIdrawer}\Model {ModelNum}\{avg_type}\fpr.npy')

    # Load thresholds
    thresholds = np.load(rf'ROC results\{ROIdrawer}\Model {ModelNum}\{avg_type}\thresholds.npy')
    
    # Find Youden indx
    Yindx = np.where( (tpr-fpr)== np.max(tpr-fpr))[0][0]
    threshold = (thresholds[Yindx])
    sensitivity = (tpr[Yindx])
    specificity = 1 - (fpr[Yindx])

    # # Find threshold for 90% sensitivity
    # indx = np.sum(tpr<=0.9)
    # threshold = 0.39#0.5*(thresholds[indx-1]+thresholds[indx])
    # specificity = 1 - 0.5*(fpr[indx-1]+fpr[indx])
    
    print(threshold)
    print(sensitivity)
    print(specificity)
    
    # == At this threshold, find...
    
    # ... Number of true positives
    NTP = np.sum( (ResultDF['Average_lesion_fIC'].values >= threshold )*(ResultDF['Biopsy_Result'].values ) )
    
    # ... Number of false positives
    NFP = np.sum( (ResultDF['Average_lesion_fIC'].values >= threshold )*(1 - ResultDF['Biopsy_Result'].values ) )

    # ... Number of true negatives
    NTN = np.sum( (ResultDF['Average_lesion_fIC'].values <= threshold )*(1 - ResultDF['Biopsy_Result'].values ) )
    
    # ... Number of false negatives
    NFN = np.sum( (ResultDF['Average_lesion_fIC'].values <= threshold )*(ResultDF['Biopsy_Result'].values ) )

    print(NTP, NFP, NTN, NFN)

# confusionTable('NT', 2, 'median')


# def contour_areas(
#     PatNum, 
#     ModelNum,
#     ROIName = 'L1_b3000_NT',
#     parameter = 'fIC',
#     VERDICT_output_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\PhD_Year_1\ISMRM Submission\VERDICT outputs",
#     ROI_path = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\PhD_Year_1\ISMRM Submission\ROIs"):
    
#     # Function to experiment with contour drawing idea
    
#     # First, read in fIC map and mask
   
#     fIC = loadmat(f'{VERDICT_output_path}/{PatNum}/Model {ModelNum}/{parameter}.mat')[parameter]
    
#     # remove infinities
#     fIC[fIC == np.inf] = 0
    
#     # Remove nan
#     fIC[np.isnan(fIC)] = 0
    
#     # Permute array axes (account for MATLAB Python differences)
#     fIC = np.moveaxis( fIC , -1, 0)   
    
#     ROIMask = np.load(f'{ROI_path}/{PatNum}/{ROIName}.npy')
    
#     # Mulitply mask and fIC
#     ROI = fIC*ROIMask
    
#     sliceIndx = np.where(np.sum(ROI, axis = (1,2)) != 0)[0][0]
    
#     ROIslice = ROI[sliceIndx]
    
#     # Smooth
#     ROIslice = scipy.ndimage.gaussian_filter(ROIslice, 1)
    
#     # plt.figure
#     # plt.imshow(ROI[sliceIndx])
#     # plt.show()
    
    
#     # Levels
#     levels = np.arange(0,1, 0.02)
#     Areas = []
#     for level in levels:
        
#         contours =  skimage.measure.find_contours(ROIslice, level)
        
        
#         # Choose largest contour
#         try:
#             lens = np.array([len(cont) for cont in contours])
#             maxindx = np.where(lens == np.max(lens))[0][0]
#             contour = contours[maxindx]
        
#         except:
#             Area = 0
#             Areas.append(Area)
#             continue
        
#         # Make path
#         cont_path = mpltPath.Path(contour)
        
        
#         # For efficiency, only query slice coordinates near contour
#         cont_xmin = np.min( contour[...,0] ) - 1
#         cont_xmax = np.max( contour[...,0] ) + 1
#         cont_ymin = np.min( contour[...,1] ) - 1
#         cont_ymax = np.max( contour[...,1] ) + 1
        
#         # Construct coordinate array
#         dx = 0.1
#         xs = np.arange(cont_xmin, cont_xmax, dx)
#         ys = np.arange(cont_ymin, cont_ymax, dx)
   
#         Ys, Xs = np.meshgrid(ys, xs, indexing = 'ij')
#         Coords = np.stack((Xs, Ys), axis = -1)

#         Area = 0
        
#         # Query each point
#         for xindx in range(len(xs)):
#             for yindx in range(len(ys)):
                
#                 point = Coords[yindx, xindx,:]
                
#                 if cont_path.contains_point(point):
#                     Area += dx**2
             
#         # print(Area)   
#         Areas.append(Area)     

#         # fig, ax = plt.subplots()
#         # ax.imshow(ROIslice, cmap=plt.cm.gray)

#         # for contour in contours:
#         #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


#         # ax.axis('image')
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         # plt.show(
#     # Integrate area under graph
#     if Areas[0]!=0:
#         Areas = np.array(Areas)/Areas[0]   
#     else:
#         Areas = np.array(Areas)
        
#     Score = scipy.integrate.trapezoid(Areas*levels, levels)
    
#     return Score
    
#     # plt.figure()
#     # plt.plot(levels, Areas)
#     # plt.show()
    
    

