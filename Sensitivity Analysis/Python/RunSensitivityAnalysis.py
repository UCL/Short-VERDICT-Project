# Python function (Callable from MATLAB) to run analysis on diagnostic sensitivity


import sys
import os
import numpy as np

# Import fIC analysis functions
sys.path.insert(0,r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Sensitivity Analysis\Python")
import sensitivity #type: ignore

# Import ROIs function
sys.path.insert(0, r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\ROIs")
import ROI # type: ignore




def RunSensitivityAnalysis(
    PatNums,
    modeltypes,
    schemenames,
    fittingtechniques,
    parameters = 'fIC',
    scoretype = 'median',
    classifier = 'threshold',
    noisetype = 'Rice',
    sigma0train = 0.05,
    T2train=10000,
    datafolder = '',
    ROIdrawer = 'NT',
    ROIname = 'L1_b3000_NT',
    ROIfolder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\ROIs"
    
):
    
    
    T2train = int(T2train)
    

    '''
    INPUTS
    ¯¯¯¯¯¯
    > PatNums: list of patient numbers (str) to be included in analysis
    > modeltypes: string or list of strings defining model types to be analysed
    > schemenames: string or list of strings defining schemes to be analysed
    > fittingtechniques: string or list of strings defining fitting techniques to be analysed
    
    *input arrays for above inputs must be the same shape
    
    > parameters: parameter(s) used for lesion characterisation (str or array of str)
    > classifier: how parameters are used for characterisation options: 'threshold', 'MLP'
    > sigma0train: noise magnitude used when generating MLP training data
    > datafolder: folder in which VERDICT outputs are stored (str)
    > ROIdrawer: initials of ROI drawer (str)
    > ROIname: name of ROI (str)
    > ROIfolder: folder in which ROIs are stored
    
    
    OUTPUTS
    ¯¯¯¯¯¯¯
    > ResultsDF: dataframe of sensitivity results
    
    '''
    
    # == Create list of modeltype, schemename, fittingtechnique combinations for analysis
    
    PatNums = list(PatNums)
    modeltypes = list(modeltypes)
    schemenames = list(schemenames)
    fittingtechniques = list(fittingtechniques)
    parameters = list(parameters)

    # print(PatNums)
    # print(modeltypes)

    # if type(modeltypes) == str:
    #     modeltypes = [modeltypes]
    # if type(schemenames) == str:
    #     schemenames = [schemenames]       
    # if type(fittingtechniques) == str:
    #     fittingtechniques = [fittingtechniques] 
        
    # Protocol combinations         
    combos = []
               
    for indx, modeltype in enumerate(modeltypes):
        combos.append([modeltypes[indx], schemenames[indx], fittingtechniques[indx]])
    

    # == Loop over combinations
    for index, combo in enumerate(combos):
        
        modeltype = combo[0]
        schemename = combo[1]
        fittingtechnique = combo[2]
        
    
        
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

                # print(f'Loading ROI mask for... {PatNum}')
                goodPatNums.append(PatNum)
                
            except:
                
                # print(f'Saving ROI mask for... {PatNum}')
                
                try:
                    ROI.saveROImask(
                        PatNum,
                        ROIdrawer,
                        ROIname,
                        datafolder = datafolder,
                        ROIfolder = ROIfolder
                        )
                    goodPatNums.append(PatNum)
                except:
                    print(f'No ROI for {PatNum}')
                

        print(len(goodPatNums))
        
        # Extract parameter values from ROIs
        for PatNum in goodPatNums:
            
            sensitivity.extractROIvalues(
                                    PatNum, 
                                    ROIdrawer = ROIdrawer,
                                    ROIName = ROIname, 
                                    modeltype = modeltype,
                                    schemename = schemename,
                                    fittingtechnique = fittingtechnique,
                                    noisetype = noisetype,
                                    sigma0train = sigma0train,
                                    T2train = T2train,
                                    parameters = parameters,
                                    MaskType = 'numpy',
                                    VERDICT_output_path = rf"{datafolder}\New VERDICT outputs",
                                    ROI_path = f'{ROIfolder}/{ROIdrawer}',
                                    output_path = rf"{datafolder}\ROI Parameter Values"
                                    )
            
        
        # Only carry forward patients with ROI
        PatNums = goodPatNums
        
        
        results_path = f'{datafolder}/Biopsy Results'
        # == Read biopsy results
        sensitivity.readBiopsyResults(
            biopsy_data_xlsx = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\INNOVATE\Biopsy Results.xlsx",
            results_path = results_path
            )
        
        
        # == If scoring method requires model training, do this here
        
        # if scoretype == 'perceptron':
            
            
        #     sensitivity.trainPerceptron(
        #         PatNums,
        #         parameters,
        #         datafolder = datafolder,
        #         ROIdrawer = ROIdrawer,
        #         ROIName = ROIname, 
        #         modeltype = modeltype,
        #         schemename = schemename,
        #         fittingtechnique = fittingtechnique,
        #         sigma0train = sigma0train,
        #         )
            
             
        
        # == Calculate ROI scores (this function should load and call trained perceptron)
        sensitivity.ROIscore(
            PatNums = PatNums,
            modeltype=modeltype,
            schemename = schemename,
            fittingtechnique = fittingtechnique,
            noisetype = noisetype,
            sigma0train = sigma0train,
            T2train = T2train,
            ROIdrawer = ROIdrawer,
            ROIName = ROIname,
            parameters = parameters,
            scoretype = scoretype,
            datafolder = datafolder,
            data_path = rf"{datafolder}\ROI Parameter Values",
            results_path = rf"{datafolder}\ROI Scores",
        )           
        
      
    
    
        
# RunSensitivityAnalysis(['BAR_003', 'BAR_033', 'INN_175', 'INN_145', 'INN_209', 'BAR_010','INN_241'], ['Original VERDICT'], ['Original Full'], ['MLP'], ['fIC', 'T2'], scoretype = 'perceptron')

    
# datafolder = str(open(datafolder, 'r').read())    
    
RunSensitivityAnalysis(
    PatNums,
    modeltypes,
    schemenames,
    fittingtechniques,
    parameters,
    scoretype,
    classifier,
    noisetype,
    sigma0train,
    T2train,
    datafolder, 
    ROIdrawer,
    ROIname,
    ROIfolder
)   