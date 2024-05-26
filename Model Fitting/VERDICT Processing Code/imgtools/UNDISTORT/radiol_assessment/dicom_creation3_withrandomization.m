clear all;

%%MATLAB reconstructions are available at
%%D:\Patient_Reconstructions\Patient2\noSENSE\b0 for b0
%%D:\Patient_Reconstructions\Patient2\noSENSE\nob0 for b>0

main_path='D:\Dicom_creation_randomization_and_decoding\';
patient='Patient14\';
dicom_patient='';%'P2' for Patient 2
t2w_im_std=load(strcat(main_path,patient,'\t2w_p14_std.mat'));
t2w_im_std=circshift(t2w_im_std.t2w_p14_std,[0 0]);
header.PatientName.FamilyName='Patient14';
patient_nr=14;

% islice=load(strcat('D:\Patients_Data_kspace\',patient,'noSENSE\slice_number'));
% islice=islice.slice_number;%slice number as whole volume is here

islice=7;


dicom_t2w_std='T2W_STD';
methods{1}='noSENSE\';
bvalues{1}='b0';

methods{2}='noSENSE\';
bvalues{2}='nob0';

%methods{3}='SENSE\';
%bvalues{3}='b0';

%methods{4}='SENSE\';
%bvalues{4}='nob0';

destination_main_path='D:\Dicom_creation_randomization_and_decoding\Dicoms\'; %% dicoms are created at this destination

header.StudyInstanceUID=strcat('1.3.6.1.4.1.5962.99.1.1039313316.355774179.1504277899684.2630.',num2str(patient_nr));

itr1=1;

array1_b0_order=[2:7];
array1_b0_order_randomized=array1_b0_order(randperm(length(array1_b0_order)));

array2_b500_order=[8:13];
array2_b500_order_randomized=array2_b500_order(randperm(length(array2_b500_order)));

array_order_randomized=[array1_b0_order_randomized(:);array2_b500_order_randomized(:)];

%save array_order_randomized array_order_randomized
    image_mat=abs(t2w_im_std)/max(abs(t2w_im_std(:)));
    dicom_name=strcat(dicom_t2w_std,'.dcm');
    header.SeriesNumber=itr1;
    header.SeriesDescription=dicom_name;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    header.SeriesNumber=itr1;
    header.SeriesTime=itr1;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient)
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);

     itr1=1;


for ii=1:2;
    
    method=methods{ii};
    bvalue=bvalues{ii};

% method='SENSE\';%'SENSE' or 'noSENSE'
% bvalue='nob0';%'b0' or 'nob0'



folder_path=(strcat(main_path,patient,method,bvalue));


topup_rec_path='D:\Dicom_creation_randomization_and_decoding\Patient14_Topup\';
topup_folder_path=strcat(topup_rec_path,patient,method,bvalue);


    




    dicom_sense='SENSEOff_'
if strcmp(bvalue,'b0')
    dicom_b='_b0_'
    load(strcat(folder_path,'/rec_nocorr_b0_UP_noSENSE.mat'));
    load(strcat(folder_path,'/rec_nocorr_b0_DN_noSENSE.mat'));
    load(strcat(folder_path,'/rec_corr_b0_UP_noSENSE.mat'));
    load(strcat(folder_path,'/rec_corr_b0_DN_noSENSE.mat'));
    load(strcat(folder_path,'/rec_corr_b0_both_noSENSE.mat'));
    load(strcat(topup_folder_path,'/rec_b0.mat'));
    
    itr=array_order_randomized(itr1);
    image_mat=abs(rec_nocorr_b0_UP_noSENSE);
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
    itr1=itr1+1;
    
     itr=array_order_randomized(itr1);
    image_mat=abs(rec_nocorr_b0_DN_noSENSE);
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
    itr1=itr1+1;
    
    itr=array_order_randomized(itr1);
    image_mat=abs(rec_corr_b0_UP_noSENSE)/sqrt(size(rec_corr_b0_UP_noSENSE,1)*size(rec_corr_b0_UP_noSENSE,2));
   dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
    itr1=itr1+1;
    
    itr=array_order_randomized(itr1);
    image_mat=abs(rec_corr_b0_DN_noSENSE)/sqrt(size(rec_corr_b0_DN_noSENSE,1)*size(rec_corr_b0_DN_noSENSE,2));;
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
    itr1=itr1+1;
    
    itr=array_order_randomized(itr1);
    image_mat=abs(rec_corr_b0_both_noSENSE)/sqrt(size(rec_corr_b0_both_noSENSE,1)*size(rec_corr_b0_UP_noSENSE,2));;
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
     header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
      itr1=itr1+1;
    
      itr=array_order_randomized(itr1);
    image_mat=abs(rec_b0(:,:,islice,1));
   dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
     header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
   
      itr1=itr1+1;

else
     dicom_b='_b500_'
    load(strcat(folder_path,'/rec_nocorr_bnot0_UP_noSENSE.mat'));
    load(strcat(folder_path,'/rec_nocorr_bnot0_DN_noSENSE.mat'));
    load(strcat(folder_path,'/rec_corr_bnot0_UP_noSENSE.mat'));
    load(strcat(folder_path,'/rec_corr_bnot0_DN_noSENSE.mat'));
    load(strcat(folder_path,'/rec_corr_bnot0_both_noSENSE.mat'));
     load(strcat(topup_folder_path,'/rec_bnot0.mat'));
    
   
    itr=array_order_randomized(itr1);
    image_mat=mean(mean(abs(rec_nocorr_bnot0_UP_noSENSE),3),4);
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
      itr1=itr1+1;
    
       itr=array_order_randomized(itr1);
    image_mat=mean(mean(abs(rec_nocorr_bnot0_DN_noSENSE),3),4);
   dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
   header.SeriesDescription=dicom_name;
   header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient);
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
      itr1=itr1+1;
    
       itr=array_order_randomized(itr1);
    image_mat=mean(mean(abs(rec_corr_bnot0_UP_noSENSE),3),4)/sqrt(size(rec_corr_bnot0_UP_noSENSE,1)*size(rec_corr_bnot0_UP_noSENSE,2));;;
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient)
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
      itr1=itr1+1;
    
       itr=array_order_randomized(itr1);
    image_mat=mean(mean(abs(rec_corr_bnot0_DN_noSENSE),3),4)/sqrt(size(rec_corr_bnot0_UP_noSENSE,1)*size(rec_corr_bnot0_UP_noSENSE,2));;;
   dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient)
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
      itr1=itr1+1;
    
       itr=array_order_randomized(itr1);
    image_mat=mean(mean(abs(rec_corr_bnot0_both_noSENSE),3),4)/sqrt(size(rec_corr_bnot0_UP_noSENSE,1)*size(rec_corr_bnot0_UP_noSENSE,2));;;;
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
    header.SeriesDescription=dicom_name;
    header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient)
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
    
      itr1=itr1+1;
    
       itr=array_order_randomized(itr1);
    image_mat=mean(mean(abs(squeeze(rec_bnot0(:,:,islice,:,:))),3),4);
    dicom_name=strcat(dicom_sense,dicom_b,num2str(itr),'.dcm');
   header.SeriesDescription=dicom_name;
   header.SeriesNumber=itr;
    image_mat=image_mat(:,size(image_mat,2)/4+1:size(image_mat,2)*3/4);
    header.Width=size(image_mat,1);
    header.Height=size(image_mat,2);
    header.Rows=size(image_mat,1);
    header.Columns=size(image_mat,2);
    header.SliceThickness=4;header.InstanceNumber=1;
    header.SpacingBetweenSlices=0;
    header.SliceLocation=(header.InstanceNumber-1)*header.SliceThickness;
    %header.SeriesNumber=1;
    header.SeriesTime=itr;
    header.PixelSpacing=[2; 2];
    header.ImagePositionPatient=[-size(image_mat,1)/2; header.SliceLocation; size(image_mat,2)/2];
    %data = uint16(image_mat);
    data = (image_mat);
    dicom_path=strcat(destination_main_path,patient)
   % dicomwrite(data, [ 'D:/dicoms_gastao_diastole_bh/IM-' sprintf('%04d',itr) '.dcm' ],'WritePrivate', true, header);
    dicomwrite(data, strcat(dicom_path,dicom_name),'WritePrivate', true, header);
    
      itr1=itr1+1;
   
    
    
end



end



    