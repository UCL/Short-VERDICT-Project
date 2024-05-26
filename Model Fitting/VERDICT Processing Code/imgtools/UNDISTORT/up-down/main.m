
%%% Main function%%%
%%% performs model based distortion correction using data from both
%%% EPI PE/A and EPI PE/P scans, potential motion between the two scans 
%%% is also compensated
%%% B0 map calculated from dual echo gradient echo B0 scan


%%% Survey and Ref scans needed
%%% Additional Toolboxes needed: MEDI toolbox,NIFTI, NiftyReg

%%% Files required
%%% PE_A/DOWN EPI scan: raw, lab files
%%% PE_P/UP EPI scan: raw, lab files
%%% Survey scan: raw, lab files
%%% SENSE reference scan: lab, raw files
%%% B0 scan: lab, raw files

clear all;
folder = fileparts('D:\Demo_code\'); 
addpath(genpath(folder));

%%% The reference is the direction PE/A (DOWN) or PE/P (UP) that has more stretching
%%% features, used in phase correction and motion matrix transformation

reference='DOWN';%%% reference direction is PE/A for this patient
islice=11; %% specific slice number to work on
N_bvalues=2;%% only do for b-value of 500 s/mm2 in addition to b-value of 0 s/mm2





MR_DOWN=MRecon('D:\Demo_code\20170630_115639_SE_DWI_0_500_PE_A_SENSEON_HalfScanOn_2mmRes_B0MapDE.lab');
MR_UP=MRecon('D:\Demo_code\20170630_115737_SE_DWI_0_500_PE_P_SENSEON_HalfScanOn_2mmRes_B0MapDE.lab');
MR_dB0=MRecon('D:\Demo_code\20170630_115334_B0map_2mmIsoRes_DE.lab');
ref=MRecon('D:\Demo_code\20170630_113159_SenseRefScan.lab');
coilsurvey=MRecon('D:\Demo_code\20170630_112927_CoilSurveyScan.lab');



S = MRsense( ref, MR_DOWN, coilsurvey );S.Smooth=1;
S.Mask=1;
S.Extrapolate=1;
 MR_DOWN.Parameter.Recon.SENSERegStrength=1e-3;S.MatchTargetSize=1;
S.Perform;
MR_DOWN.Parameter.Recon.Sensitivities = S;
MR_DOWN.Parameter.Parameter2Read.typ = 1;
MR_DOWN.ReadData;
MR_DOWN.RandomPhaseCorrection;
MR_DOWN.PDACorrection;
MR_DOWN.DcOffsetCorrection;
MR_DOWN.MeasPhaseCorrection;
MR_DOWN.SortData;
MR_DOWN.GridData;
%%% RingingFilterStrength variable between patients
MR_DOWN.Parameter.Recon.RingingFilterStrength=[0.25 0.25 0.25];
MR_DOWN.Parameter.Recon.RingingFilterStrength=[1 1 1];
MR_DOWN.RingingFilter;MR_DOWN.Parameter.Recon.EPICorrectionMethod='nonlin';MR_DOWN.Parameter.Recon.EPI2DCorr='yes';
%MR_DOWN.ZeroFill;
MR_DOWN.K2IM;
MR_DOWN.EPIPhaseCorrection;
MR_DOWN.K2IP;
MR_DOWN.GridderNormalization;
%%% Doing SENSE first to decrease the computational load later
MR_DOWN.SENSEUnfold;
%%% PartialFourier always after SENSEunfold to retain phase
MR_DOWN.PartialFourier;
MR_DOWN.ConcomitantFieldCorrection;
MR_DOWN.DivideFlowSegments;
MR_DOWN.FlowPhaseCorrection;
MR_DOWN.RotateImage;
%%% -1 pixel shift along PE to align up and down
%%% verified from Agar phantom data acquired with Torben
MR_DOWN.Data=circshift(MR_DOWN.Data,[-1  0 0 0 0 0 0 0 0 0 0]);
MR_DOWN.I2K;




S = MRsense( ref, MR_UP, coilsurvey );S.Smooth=1;S = MRsense( ref, MR_UP, coilsurvey );S.Smooth=1;
S.Mask=1;
S.Extrapolate=1;
 MR_UP.Parameter.Recon.SENSERegStrength=1e-3;S.MatchTargetSize=1;
S.Perform;
MR_UP.Parameter.Recon.Sensitivities = S;
MR_UP.Parameter.Parameter2Read.typ = 1;
MR_UP.ReadData;
MR_UP.RandomPhaseCorrection;
%MR_UP.RemoveOversampling;
MR_UP.PDACorrection;
MR_UP.DcOffsetCorrection;
MR_UP.MeasPhaseCorrection;
MR_UP.SortData;
MR_UP.GridData;
MR_UP.Parameter.Recon.RingingFilterStrength=[0.25 0.25 0.25];
MR_UP.Parameter.Recon.RingingFilterStrength=[1 1 1];
MR_UP.RingingFilter;MR_UP.Parameter.Recon.EPICorrectionMethod='nonlin';MR_UP.Parameter.Recon.EPI2DCorr='yes';
%MR_UP.ZeroFill;
MR_UP.K2IM;
MR_UP.EPIPhaseCorrection;
MR_UP.K2IP;
MR_UP.GridderNormalization;
MR_UP.SENSEUnfold;
MR_UP.PartialFourier;
MR_UP.ConcomitantFieldCorrection;
MR_UP.DivideFlowSegments;
%MR_UP.GeometryCorrection;
%MR_UP.RemoveOversampling;
%MR_UP.ZeroFill;
MR_UP.FlowPhaseCorrection;
MR_UP.RotateImage;
%MR_UP.Data=circshift(MR_UP.Data,[1 1 0 0 0 0 0 0 0 0 0 0]);
%% 1 pixel shift along PE and FE
%%% verified from Agar phantom data acquired with Torben
MR_UP.Data=circshift(MR_UP.Data,[1 1 0 0 0 0 0 0 0 0 0 0]);
MR_UP.I2K;


kspace_up=double(MR_UP.Data); %%% direction UP/PE-P
kspace_down=double(MR_DOWN.Data);%%%direction DOWN/PE-A


%%%% B0 scan

S = MRsense( ref, MR_dB0, coilsurvey );S.Smooth=1;
S.Mask=1;
S.Extrapolate=1;S.MatchTargetSize=1;
S.Perform;
 MR_dB0.Parameter.Recon.SENSERegStrength=1e-3;
 MR_dB0.Parameter.Recon.Sensitivities = S;
MR_dB0.Parameter.Parameter2Read.typ = 1;
MR_dB0.ReadData;
MR_dB0.RandomPhaseCorrection;
%MR_dB0.RemoveOversampling;
MR_dB0.PDACorrection;
MR_dB0.DcOffsetCorrection;
MR_dB0.MeasPhaseCorrection;
MR_dB0.SortData;
MR_dB0.GridData;
MR_dB0.Parameter.Recon.RingingFilterStrength=[1 1 1];
MR_dB0.RingingFilter;
MR_dB0.ZeroFill;
MR_dB0.K2I;
%MR_dB0.Data=circshift(MR_dB0.Data,[1 1 -1 0 0 0 0 0 0 0 0 0]);
MR_dB0.GridderNormalization;
MR_dB0.SENSEUnfold;
MR_dB0.PartialFourier;
MR_dB0.ConcomitantFieldCorrection;
MR_dB0.DivideFlowSegments;
%MR_dB0.GeometryCorrection;
%MR_dB0.RemoveOversampling;
%MR_dB0.ZeroFill;
MR_dB0.FlowPhaseCorrection;
MR_dB0.RotateImage;

c1=squeeze(double(MR_dB0.Data));
B0_image_data_allcoils=c1;
c1=permute(c1,[1 2 3 5 4]);
%%% B0 phase estimation from multiple echoes, last dimension is always
%%% echoes
[iFreq_raw3 N_std] = Fit_ppm_complex(squeeze(c1));
%%% magnitude mask used sometimes where there rectal air size is big
iMag=squeeze(double(sqrt(sum(abs(MR_dB0.Data(:,:,:,:,1,1,1,1,1,1)).^2,4))));
%mask=Jmask(Normalize(iMag,0,1),0.05);
%%% PDF estimation not used
%RDF2= PDF2(iFreq_raw3,N_std,mask,size(mask),MR_dB0.Parameter.Scan.RecVoxelSize, [0 0 1],1e-6);
RDF2=double(iFreq_raw3);
%%% divide B0 phase by TE to get B0 Hz
dB0_data=RDF2/(2*pi*2.3e-3);



%%%% Transform B0 to EPI speace 

MR_scan=MR_UP;
MR_dB0=MR_dB0;
A = MR_scan.Transform('ijk', 'RAF',1 );
B = MR_dB0.Transform('ijk', 'RAF', 1 );
[ x,y,z ] = ndgrid( 1:size(MR_scan.Data,1), 1:size(MR_scan.Data,2), 1:size(MR_scan.Data,8) );
X = [x(:), y(:), z(:), ones( size(x(:)))]';
Xref = inv(B)*A*X;


dB0_data=dB0_data;
%%% interpolate dB0 to EPI DOWN scan coordinates
ref_slice = interp3( dB0_data, ...
Xref(2,:), Xref(1,:), Xref(3,:) );
ref_slice = reshape( ref_slice, [size(MR_scan.Data,1) size(MR_scan.Data,2) size(MR_scan.Data,8) ]);
dB0=ref_slice;
dB0(isnan(dB0))=0;



%%% simulate Cartesian trajectory
opt.sx=size(kspace_up,2);opt.sy=size(kspace_up,1);
[tr.x tr.y]=createSim_EPItraj(opt.sy,opt.sx);



%%% creating timing matrix, contains sampling time for each sample in k-space matrix
row=size(kspace_up,1);
col=size(kspace_up,2);
fieldstrength_tesla=3;
%epifactor=size(MR_UP.Data,2);
%epifactor=MR_UP.Parameter.Scan.Samples(2);
water_fat_diff_ppm          = 3.4;  
resonance_freq_mhz_tesla    = 42.576; % gyromagnetic ratio for proton (1H)
%echo_train_length           = epifactor + 1 ;
water_fat_shift_hz          = fieldstrength_tesla * water_fat_diff_ppm * resonance_freq_mhz_tesla % water_fat_shift_hz 3T = 434.215 Hz
water_fat_shift=MR_UP.Parameter.GetValue('UGN5_RFE_act_water_fat_shift'); %25; %25
echo_spacing=water_fat_shift(1)/(water_fat_shift_hz*(size(MR_UP.Data,1)+1));

%%% the other way to get echo spacing
% MR_UP.Parameter.ExtractPDFFile('file.pdf');
% c1=1e-3*diff_time;
% echo_spacing2=c1(2)-c1(1);

%%%% timing matrix

temp=[0:1:row-1]';
temp=temp-row/2;
temp=temp*echo_spacing;
t_s_m1=repmat(temp,[1 col]);


% temp=[1:col];
% temp=temp-ceil(col/2);
% temp=temp*echo_spacing;
% t_s_m1=repmat(temp,[row 1]);

 
kspace_up=double(MR_UP.Data);kspace_down=double(MR_DOWN.Data);


bw_per_pixel=MR_UP.Parameter.GetValue('UGN12_ACQ_act_bw_per_pixel');
bw_per_pixel=bw_per_pixel(1);
acq_voxel_size=MR_DOWN.Parameter.Scan.AcqVoxelSize;

%%% Step 1: Find B0 frequency offset, scalar constant in Hz
Sk1=double(kspace_up(:,:,:,1,1,1,1,islice,1,1,1,1)); %% up data
Sk2=double(kspace_down(:,:,:,1,1,1,1,islice,1,1,1,1));%% down data
dB02=smoothn(dB0(:,:,islice),2,'Robust');
arg11=struct('Sk1',squeeze(Sk1),'Sk2',squeeze(Sk2),'S',ones(size(Sk1,1),size(Sk1,2)),'t_s1',(t_s_m1),'t_s2',flipud(t_s_m1),'dB0_rads',dB02,'k',tr,'bw_per_pixel',bw_per_pixel(1));
fun = @(beta)afun11(beta,arg11);%afun6
options = optimset('Display','iter','PlotFcns',@optimplotfval);
options.MaxIter=10;
offset = fminsearchbnd(fun,0,-100,100,options);%%% offset in Hz found by optimization 
slice_shifts(islice)=0;
dB02=dB02+offset; %%updated B0 with frequency offset

%%% reconstruction in UP and DOWN using updated B0
arg1=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',-slice_shifts(islice));
arg2=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',flipud(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',slice_shifts(islice));
corrected_offsetandB0_UP(:,:,islice)= Eh_v7(Sk1,arg1);
corrected_offsetandB0_DOWN(:,:,islice)= Eh_v7(Sk2,arg2);

%%% intensity correction suggested in Munger dependent on gradient in B0
%%% along PE direction, may be redundant
pixel_shift=(dB02/bw_per_pixel);
for ii=3:size(Sk1,1)-2;
    for jj=1:size(Sk1,2);
temp=[pixel_shift(ii-2,jj) pixel_shift(ii-1,jj) pixel_shift(ii,jj) pixel_shift(ii+1,jj) pixel_shift(ii+2,jj)];
p=polyfit([1:5],temp,1);p=p(1);
corrected_offsetandB0_UP(ii,jj,islice)=(1+p)*corrected_offsetandB0_UP(ii,jj,islice);
corrected_offsetandB0_DOWN(ii,jj,islice)=(1-p)*corrected_offsetandB0_DOWN(ii,jj,islice);
end;end


%%% Find motion matrix to account for any potential motion between UP and
%%% DOWN scans using NiftyReg
%%% UP or DOWN scan will be reference 
%%% important is to select the reference as the one with stretching artifact
%%% motion matrix will only be applied to one direction UP/DOWN, reference
%%% will remain unchanged


%%% Creating nifi files for NiftyReg
im1=corrected_offsetandB0_UP(:,:,islice);im11=Normalize(im1,0,1);
im2=corrected_offsetandB0_DOWN(:,:,islice);im22=Normalize(im2,0,1);
clear imnii1;imnii1 = make_nii(abs(im11));
save_nii(imnii1,'D:\Demo_code\NIFTI\nifty_data\epi1.nii');
clear imnii1;imnii1 = make_nii(abs(im22));
save_nii(imnii1,'D:\Demo_code\NIFTI\nifty_data\epi2.nii');
niftypath='D:\Demo_code\NiftyReg\';
data_path='D:\Demo_code\NIFTI\nifty_data\';
epi='epi1.nii ';
t2w='epi2.nii ';

if strcmp(reference,'DOWN');
    %%% create motion matrix by doing registrations using NiftyReg
interpolation_matrix1=calculate_interpolation_matrix(niftypath,data_path,epi, t2w);
%%% new operator containing motion matrix
arg1=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',-slice_shifts(islice),'motion_matrix',interpolation_matrix1,'bw_per_pixel',bw_per_pixel(1));
corrected_image_UP2(:,:,islice)= Eh_v7_motionmatrix(Sk1,arg1);
arg2=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',flipud(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',slice_shifts(islice),'motion_matrix',eye(size(interpolation_matrix1)),'bw_per_pixel',bw_per_pixel(1));
corrected_image_DOWN2(:,:,islice)= Eh_v7_motionmatrix(Sk2,arg2);

else 
     %%% create motion matrix by doing registrations using NiftyReg
interpolation_matrix1=calculate_interpolation_matrix(niftypath,data_path,t2w, epi);
%%% new operator containing motion matrix
arg1=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',-slice_shifts(islice),'motion_matrix',eye(size(interpolation_matrix1)),'bw_per_pixel',bw_per_pixel(1));
corrected_image_UP2(:,:,islice)= Eh_v7_motionmatrix(Sk1,arg1);
arg2=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',flipud(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',slice_shifts(islice),'motion_matrix',interpolation_matrix1,'bw_per_pixel',bw_per_pixel(1));
corrected_image_DOWN2(:,:,islice)= Eh_v7_motionmatrix(Sk2,arg2);
end


%%% Model based reconstruction for b-value of 0
opt.num_it=5;
opt.r_init=zeros(opt.sy,opt.sx);

Sk=[Sk1;Sk2];
opt.recon_choice='cg';   % 'Conjugate Gradient Reconstruction'
opt.Eh_fh = @Eh_v5_motionmatrixoffset;
opt.E_fh  = @E_v5_motionmatrixoffset;
I_undis1=Recon_withdB0_USM4_motionmatrixoffset(Sk,opt,arg1,arg2);
rec(:,:,islice)=I_undis1(:,:,end);

rec_nocorr_b0_UP_SENSE=fftshift(ifft2(ifftshift(Sk1))); %%% no correction direction UP
rec_nocorr_b0_DN_SENSE=fftshift(ifft2(ifftshift(Sk2))); %%% no correction direction DOWN
rec_corr_b0_UP_SENSE=(corrected_image_UP2(:,:,islice)); %%% corrected direction UP only, preliminary reconstruction
rec_corr_b0_DN_SENSE=(corrected_image_DOWN2(:,:,islice));%%% corrected direction DOWN only, preliminary reconstruction
rec_corr_b0_both_SENSE=rec(:,:,islice); %%% Model based reconstruction using both UP and DOWN data

save rec_nocorr_b0_UP_SENSE rec_nocorr_b0_UP_SENSE
save rec_nocorr_b0_DN_SENSE rec_nocorr_b0_DN_SENSE
save rec_corr_b0_UP_SENSE rec_corr_b0_UP_SENSE
save rec_corr_b0_DN_SENSE rec_corr_b0_DN_SENSE
save rec_corr_b0_both_SENSE rec_corr_b0_both_SENSE

N_bdir=size(kspace_up,11); % number of diffusion directions
N_avg=size(kspace_up,12);%number of averages



for b_dir=1:N_bdir;for avg=1:N_avg;
Sk1=double(kspace_up(:,:,:,1,1,1,1,islice,1,N_bvalues,b_dir,avg));
Sk2=double(kspace_down(:,:,:,1,1,1,1,islice,1,N_bvalues,b_dir,avg));
%%% uncorrected reconstructions
rec_nocorr_bnot0_UP_SENSE(:,:,b_dir,avg)=fftshift(ifft2(ifftshift(Sk1)));%%% no correction direction UP
rec_nocorr_bnot0_DN_SENSE(:,:,b_dir,avg)=fftshift(ifft2(ifftshift(Sk2)));%%% no correction direction DOWN

% corrected reconstructions using B0 only using transpose
if strcmp(reference,'DOWN');
arg1=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',-slice_shifts(islice),'motion_matrix',interpolation_matrix1,'bw_per_pixel',bw_per_pixel(1));
corrected_image_UP2(:,:,islice)= Eh_v7_motionmatrix(Sk1,arg1);
arg2=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',flipud(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',slice_shifts(islice),'motion_matrix',eye(size(interpolation_matrix1)),'bw_per_pixel',bw_per_pixel(1));
corrected_image_DOWN2(:,:,islice)= Eh_v7_motionmatrix(Sk2,arg2);
else 
    arg1=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',-slice_shifts(islice),'motion_matrix',eye(size(interpolation_matrix1)),'bw_per_pixel',bw_per_pixel(1));
corrected_image_UP2(:,:,islice)= Eh_v7_motionmatrix(Sk1,arg1);
arg2=struct('S',ones(size(Sk1,1),size(Sk1,2)),'t_s',flipud(t_s_m1),'dB0_rads',2*pi*dB02,'k',tr,'offset',slice_shifts(islice),'motion_matrix',interpolation_matrix1,'bw_per_pixel',bw_per_pixel(1));
corrected_image_DOWN2(:,:,islice)= Eh_v7_motionmatrix(Sk2,arg2);
end

rec_corr_bnot0_UP_SENSE(:,:,b_dir,avg)=(corrected_image_UP2(:,:,islice));%%% corrected direction UP only, preliminary reconstruction
rec_corr_bnot0_DN_SENSE(:,:,b_dir,avg)=(corrected_image_DOWN2(:,:,islice));%%% corrected direction DOWN only, preliminary reconstruction

%% intensity correction as proposed in Munger
pixel_shift=(dB02/bw_per_pixel(1));
for ii=3:size(Sk1,1)-3;
    for jj=1:size(Sk1,2);
temp=[pixel_shift(ii-2,jj) pixel_shift(ii-1,jj) pixel_shift(ii,jj) pixel_shift(ii+1,jj) pixel_shift(ii+2,jj)];
p=polyfit([1:5],temp,1);p=p(1);
corrected_image_UP2(ii,jj,islice)=(1+p)*corrected_image_UP2(ii,jj,islice);
corrected_image_DOWN2(ii,jj,islice)=(1-p)*corrected_image_DOWN2(ii,jj,islice);
end;end

%% Step 2: Phase correction for b-value>0 
%% important is to select the reference as the one with stretching artifact
%% correction is not applied to reference
im1=corrected_image_UP2(:,:,islice);
im2=corrected_image_DOWN2(:,:,islice);
if strcmp(reference,'DOWN');%%%im2 is reference
angle1=angle(im1.*conj(im2));im11=fftshift(ifft2(ifftshift(Sk1)));sk11=im11.*exp(-i*angle1);
Sk11=fftshift(fft2(ifftshift(sk11)));
Sk=[Sk11;Sk2];
else %%%im1 is reference
angle1=angle(im2.*conj(im1));im22=fftshift(ifft2(ifftshift(Sk2)));sk22=im22.*exp(-i*angle1);
Sk22=fftshift(fft2(ifftshift(sk22)));
Sk=[Sk1;Sk22]; 
end

%% Step 3: Model based reconstruction
opt.num_it=3;
opt.recon_choice='cg';   % 'cg' 
opt.Eh_fh = @Eh_v5_motionmatrixoffset;
opt.E_fh  = @E_v5_motionmatrixoffset;
I_undis1=Recon_withdB0_USM4_motionmatrixoffset(Sk,opt,arg1,arg2);
rec_corr_bnot0_both_SENSE(:,:,b_dir,avg)=I_undis1(:,:,end);%%% Model based reconstruction using both UP and DOWN data
    
    end;end

save rec_nocorr_bnot0_UP_SENSE rec_nocorr_bnot0_UP_SENSE
save rec_nocorr_bnot0_DN_SENSE rec_nocorr_bnot0_DN_SENSE
save rec_corr_bnot0_UP_SENSE rec_corr_bnot0_UP_SENSE
save rec_corr_bnot0_DN_SENSE rec_corr_bnot0_DN_SENSE
save rec_corr_bnot0_both_SENSE rec_corr_bnot0_both_SENSE


