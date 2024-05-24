import sys
import glob
import os
import numpy as np
import SimpleITK as sitk

# Import DICOM from imgtools
sys.path.insert(0, r"C:\Users\adam\OneDrive - University College London\UCL PhD\Image-Processing\DICOM\Python")
import DICOM # type: ignore


# Function for saving ROI masks
def saveROImask(
                PatNum, 
                ROIdrawer,
                ROIName, 
                datafolder = r"C:\Users\adam\OneDrive\Desktop\INNOVATE STUDY COHORT VERDICT IMAGES", 
                ROIfolder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\INNOVATE\ROIs",
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
 