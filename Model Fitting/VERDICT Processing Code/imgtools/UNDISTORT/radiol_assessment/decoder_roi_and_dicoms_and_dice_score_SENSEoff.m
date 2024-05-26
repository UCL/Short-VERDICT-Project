%%%%%function
%%%% inputs:
%%%% 1)ROIs in .nii exported from Horos
%%%% 2)Images in dicoms on which ROIs were drawn
%%%% 3) array order randomization, 2-7 for b0 and 8-13 for b500
%%%% function decodes back the individual reconstrcutions with
%%%% corresponding ROI
clear all;
addpath('D:\Dicom_creation_randomization_and_decoding\NIFTI\');
patient_nr=14;
dicom_folder_path=strcat('D:\Dicom_creation_randomization_and_decoding\Dicoms\Patient',num2str(patient_nr),'\');
roi_folder_path='D:\Dicom_creation_randomization_and_decoding\Patient14_noSENSE_rois\';
load('D:\Dicom_creation_randomization_and_decoding\array_order_randomized.mat');

dice_score_b0=[];
dice_score_b500=[];

ii=1;
T2W_STD=dicomread(strcat(dicom_folder_path,'T2W_STD.dcm'));

roi=load_untouch_nii(strcat(roi_folder_path,num2str(ii),'.nii'));
roi_T2W=fliplr(imrotate(double(roi.img),-90));


ii=1;
SENSEOff__b0_UC_PEA=dicomread(strcat(dicom_folder_path,'SENSEOff__b0_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b0_UC_PEA=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b0_UC_PEA);
dice_score_b0=[dice_score_b0;score];

ii=2;
SENSEOff__b0_UC_PEP=dicomread(strcat(dicom_folder_path,'SENSEOff__b0_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b0_UC_PEP=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b0_UC_PEP);
dice_score_b0=[dice_score_b0;score];

ii=3;
SENSEOff__b0_C_PEA=dicomread(strcat(dicom_folder_path,'SENSEOff__b0_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b0_C_PEA=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b0_C_PEA);
dice_score_b0=[dice_score_b0;score];

ii=4;
SENSEOff__b0_C_PEP=dicomread(strcat(dicom_folder_path,'SENSEOff__b0_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b0_C_PEP=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b0_C_PEP);
dice_score_b0=[dice_score_b0;score];

ii=5;
SENSEOff__b0_C_PEAP=dicomread(strcat(dicom_folder_path,'SENSEOff__b0_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b0_C_PEAP=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b0_C_PEAP);
dice_score_b0=[dice_score_b0;score];

ii=6;
SENSEOff__b0_Topup=dicomread(strcat(dicom_folder_path,'SENSEOff__b0_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b0_Topup=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b0_Topup);
dice_score_b0=[dice_score_b0;score];

ii=7;
SENSEOff__b500_UC_PEA=dicomread(strcat(dicom_folder_path,'SENSEOff__b500_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b500_UC_PEA=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b500_UC_PEA);
dice_score_b500=[dice_score_b500;score];


ii=8;
SENSEOff__b500_UC_PEP=dicomread(strcat(dicom_folder_path,'SENSEOff__b500_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b500_UC_PEP=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b500_UC_PEP);
dice_score_b500=[dice_score_b500;score];


ii=9;
SENSEOff__b500_C_PEA=dicomread(strcat(dicom_folder_path,'SENSEOff__b500_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b500_C_PEA=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b500_C_PEA);
dice_score_b500=[dice_score_b500;score];


ii=10;
SENSEOff__b500_C_PEP=dicomread(strcat(dicom_folder_path,'SENSEOff__b500_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b500_C_PEP=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b500_C_PEP);
dice_score_b500=[dice_score_b500;score];

ii=11;
SENSEOff__b500_C_PEAP=dicomread(strcat(dicom_folder_path,'SENSEOff__b500_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b500_C_PEAP=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b500_C_PEAP);
dice_score_b500=[dice_score_b500;score];


ii=12;
SENSEOff__b500_Topup=dicomread(strcat(dicom_folder_path,'SENSEOff__b500_',num2str(array_order_randomized(ii)),'.dcm'));
roi=load_untouch_nii(strcat(roi_folder_path,num2str(array_order_randomized(ii)),'.nii'));
roi_SENSEOff__b500_Topup=fliplr(imrotate(double(roi.img),-90));
score=calculate_dice_score(roi_T2W,roi_SENSEOff__b500_Topup);
dice_score_b500=[dice_score_b500;score];

save dice_score_b0 dice_score_b0
save dice_score_b500 dice_score_b500

[B,L] = bwboundaries(roi_SENSEOff__b0_UC_PEA,'noholes');
figure;imshow(mat2gray(abs(SENSEOff__b0_UC_PEA)))
hold on
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
end