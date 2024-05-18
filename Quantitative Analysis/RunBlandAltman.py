# Import relevant functions
import sys
import os
import numpy as np
import pickle

# Import fIC analysis functions
sys.path.insert(0,r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Sensitivity Analysis")
import sensitivity #type: ignore


def RunBlandAltman(
    PatNums,
    modeltypes,
    schemenames,
    fittingtechniques,
    parameter = 'fIC',
    scoretype = 'median',
    sigma0train=0.05,
    datafolder = str(open(r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\output_folder.txt", 'r').read()),
    ROIdrawer = 'NT',
    ROIname = 'L1_b3000_NT',
    ROIfolder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\ROIs"
):
    
    '''
    STEPS
    
    1. Extract ROI values/scores for each patient
    '''

    PatNums = list(PatNums)
    modeltypes = list(modeltypes)
    schemenames = list(schemenames)
    fittingtechniques = list(fittingtechniques)
    try:
        sigma0trains = list(sigma0train)
    except:
        sigma0trains = [sigma0train]



    
    
    combos = []
    for i in [0,1]:
        combos.append([modeltypes[i], schemenames[i], fittingtechniques[i], sigma0trains[i]] )
                    

    protocolsROIscores = []

    # == Loop over combinations
    for index, combo in enumerate(combos):
        
        modeltype = combo[0]
        schemename = combo[1]
        fittingtechnique = combo[2]
        sigma0train = combo[3]
        
    
        
        # For each patient, check if numpy array ROI exists
        # if not, save ROI mask as numpy array. 
        # With numpy ROI masks, 
        goodPatNums = []
        for index, PatNum in enumerate(PatNums):
            
            # Try and read numpy ROI mask
            try:
                
                ROImask = np.load(
                    f'{ROIfolder}/{ROIdrawer}/numpy/{PatNum}/{ROIname}.npy'
                )

                print(f'Loading ROI mask for... {PatNum}')
                goodPatNums.append(PatNum)
                
            except:
                
                print(f'Saving ROI mask for... {PatNum}')
                
                try:
                    sensitivity.saveROImask(
                        PatNum,
                        ROIdrawer,
                        ROIname,
                        ROIfolder = ROIfolder
                        )
                    goodPatNums.append(PatNum)
                except:
                    print(f'No ROI for {PatNum}')
                

        
        # Extract parameter values from ROIs
        for PatNum in goodPatNums:
            
            sensitivity.extractROIvalues(
                                    PatNum, 
                                    ROIdrawer = ROIdrawer,
                                    ROIName = ROIname, 
                                    modeltype = modeltype,
                                    schemename = schemename,
                                    fittingtechnique = fittingtechnique,
                                    sigma0train = sigma0train,
                                    parameters = parameter,
                                    MaskType = 'numpy',
                                    VERDICT_output_path = rf"{datafolder}\VERDICT outputs",
                                    ROI_path = f'{ROIfolder}/{ROIdrawer}',
                                    output_path = rf"{datafolder}\ROI Parameter Values"
                                    )
            
        
        # == Calculate ROI scores (this function should load and call trained perceptron)
        sensitivity.ROIscore(
            PatNums = goodPatNums,
            modeltype=modeltype,
            schemename = schemename,
            fittingtechnique = fittingtechnique,
            sigma0train = sigma0train,
            ROIdrawer = ROIdrawer,
            ROIName = ROIname,
            parameters = parameter,
            scoretype = scoretype,
            datafolder = datafolder,
            data_path = rf"{datafolder}\ROI Parameter Values",
            results_path = rf"{datafolder}\ROI Scores",
        )     
        
        
        # == Load scores
        if fittingtechnique == 'MLP':
            path = f'{datafolder}\ROI Scores/{modeltype}/{schemename}/{fittingtechnique}/sigma0 = {sigma0train}/{ROIdrawer}/{ROIname}/{scoretype}'
        else:
            path = f'{datafolder}\ROI Scores/{modeltype}/{schemename}/{fittingtechnique}/{ROIdrawer}/{ROIname}/{scoretype}'
            
            
            # Save dataframe as pickle
        with open(f'{path}/scoresDF.pickle', 'rb') as handle:
            scoresDF = pickle.load(handle)
            
            
        protocolsROIscores.append(scoresDF['Score'].values)
        
    protocolsROIscores = np.asarray(protocolsROIscores)
        
    return protocolsROIscores
            
            


values = RunBlandAltman(
    PatNums,
    modeltypes,
    schemenames,
    fittingtechniques,
    parameter,
    scoretype,
    sigma0train,
    datafolder,
    ROIdrawer,
    ROIname,
    ROIfolder
)