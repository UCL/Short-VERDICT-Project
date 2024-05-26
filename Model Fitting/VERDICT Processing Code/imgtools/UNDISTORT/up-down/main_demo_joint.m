
clear all;
%%% Patient 12

%%% Joint image and B0 estimation
%%% input: raw and lab files for PE/A and PE/P scans and B0 scans
%%% outputs: 1)b-value of 0:
%%%%  rec_corr_b0_b0initial: B0 corrected reconstruction at b-value of 0
%%%%  using B0 estimated from dual echo gradient scan
%%%%  rec_nocorr_b0: reconstruction at b-value of 0 wothout any B0 correction
%%%%  rec_corr_b0: corrected reconstruction using joint image and B0 estimation framework at b-value of 0

%%%% 2)b-value>0:
%%%  rec_init: B0 corrected reconstruction at b-value> 0
%%%%  using B0 estimated from dual echo gradient scan
%%%%  rec_nocorr: reconstruction at b-value>0 wothout any B0 correction
%%%%  rec_corr: corrected reconstruction using joint image+B0 estimation framework at b-value> 0

%%% slice number for which the joint reconstruction needs to be performed
slice_number=8;
islice=8;
reference='UP'; %% reference for phase correction

folder = fileparts('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/MRecon-3.0.552/');
addpath(genpath(folder));

folder = fileparts('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/code2/');
addpath(genpath(folder));

folder = fileparts('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/functions/');
addpath(genpath(folder));

folder = fileparts('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/fessler/irt/nufft/');
addpath(genpath(folder));

folder = fileparts('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/fessler/irt/mex/');
addpath(genpath(folder));

folder = fileparts('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/fessler/irt/MEDI_toolbox2/');
addpath(genpath(folder));


folder = fileparts('/cs/research/vision/home0/green/musman/fessler (copy)/irt/fbp/'); 
addpath(genpath(folder));
 folder = fileparts('/cs/research/vision/home0/green/musman/fessler (copy)/irt/mex/'); 
 addpath(genpath(folder));
 folder = fileparts('/cs/research/vision/home0/green/musman/fessler (copy)/irt/nufft/'); 
 addpath(genpath(folder));
 addpath('/cs/research/vision/home0/green/musman/fessler (copy)/irt/utilities/');
% 
 addpath('/cs/research/vision/home0/green/musman/fessler (copy)/irt/systems/');
% 
addpath('/cs/research/vision/home0/green/musman/fessler (copy)/irt/mri/');
 addpath('/cs/research/vision/home0/green/musman/fessler (copy)/irt/wls/');
 addpath('/cs/research/vision/home0/green/musman/fessler (copy)/irt/penalty/');
 addpath('/cs/research/vision/home0/green/musman/fessler (copy)/irt/general/');



MR_DOWN=MRecon('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/i_19012018_1129396_9_2_wip_se_dwi_0_500_1000_pea_sense2_halfscanon_te55ms_senseV4.lab');
MR_UP=MRecon('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/i_19012018_1130296_10_2_wip_se_dwi_0_500_1000_pep_sense2_halfscanon_te55ms_senseV4.lab');
MR_dB0=MRecon('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/i_19012018_1123216_5_2_wip_b0map_de_3d_nosense_pe_rl_fatshifta_clearV4.lab');
ref=MRecon('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/i_19012018_1119497_1000_11_wip_senserefscan_clearV4.lab');
 coilsurvey=MRecon('/cs/research/vision/home0/green/musman/thindrives/d/Demo_code_joint/i_19012018_1119266_1000_8_wip_coilsurveyscanV4.lab');

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
MR_UP.SENSEUnfold;%%% done to make problem less computational
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
MR_dB0.SENSEUnfold; %%% done to make problem less computational
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
 

%%% create trajectory
opt.sx=size(kspace_up,2);opt.sy=size(kspace_up,1);
[tr.x tr.y]=createSim_EPItraj(opt.sy,opt.sx);


%%% creating timing matrix, contains sampling time for each sample in k-space matrix
row=size(kspace_up,1);
col=size(kspace_up,2);
fieldstrength_tesla=3;

water_fat_diff_ppm          = 3.4;  
resonance_freq_mhz_tesla    = 42.576; % gyromagnetic ratio for proton (1H)

water_fat_shift_hz          = fieldstrength_tesla * water_fat_diff_ppm * resonance_freq_mhz_tesla % water_fat_shift_hz 3T = 434.215 Hz
water_fat_shift=MR_UP.Parameter.GetValue('UGN5_RFE_act_water_fat_shift'); %25; %25
echo_spacing=water_fat_shift(1)/(water_fat_shift_hz*(size(MR4.Data,1)+1));






fmap=dB0;
kspace_up=double(kspace_up);
kspace_dn=double(kspace_down);


       %%% creating timing matrix, contains sampling time for each sample in k-space matrix
row=size(kspace_up,1);
col=size(kspace_up,2);

sense_factor=1;
temp=[0:sense_factor:row-1]';
temp=temp*echo_spacing/sense_factor;
taq=max(temp(:))-min(temp(:));
temp=temp-taq/2;
t_s_m1=repmat(temp,[1 col]);
t_s_m1=t_s_m1(:);
ti1=t_s_m1;
ti2=ti1(end:-1:1);



f.nx = row; %256
f.ny = col; %256
f.fov=[13.3 44]; 
f.fov=[row*2 col*2]/10; % should be in cm, 2mm resolution inplane

f.dr = 1; %240/256
ig = image_geom('nx',f.nx,'dx',f.fov(1)/f.nx,'ny',f.ny,'dy', f.fov(2)/f.ny,...
	'offset_x', 0.5, 'offset_y', 0.5);


mask=true(row,col);

ig.mask = mask;

N = [f.nx, f.ny];

f.te1 = 0; % echo-time 1
f.te2 = 0; % echo-time 2, no time difference at the centre of k-space
f.wi = 1;
f.chat = 1;
f.pl = 1;
%f.noise = inf; % SNR (inf is noiseless)
f.updt = 3; % # of refinements
f.out_mode = 'joint';
f.gmap_mode = 'none';
f.reg_mode = '2D';
f.tseg = 10; % time segments for nufft approach
f.basis = 'rect'; % system matrix basis ('rect' or 'dirac')
f.traj = 'cartesian'; % trajectory type
f.nufft = {N, [1 1], N, N/2, 'linear'}; %nufft parameterssize(kspa
f.ncoil=1;
f.smap = ones(row,col);%%% this can be changed if actual sensitivities used
[ks, om, witr] = mri_trajectory1(f.traj, {}, N, f.fov, {});
f.te1=0;
f.te2=0+0E-3;
ksp = [ks;ks];%%% two trajectories for UP and down scans are same
ti = [ti1;ti2+0E-3];

tmp = diag_sp(-ti(:));
	for ic=1:f.ncoil
		Tm{ic} = tmp;
	end
	T = block_fatrix(Tm, 'type', 'diag'); % [G1; G2; ... ]

traj = {ksp,ksp,1};

% Create system objects - exact for data and nufft for reconstruction
Gm = Gmri(ksp, ig.mask, 'fov', f.fov, 'basis', {f.basis}, 'nufft', f.nufft);
Gu = Gmri(ks, ig.mask, 'fov', f.fov, 'basis', {f.basis}, 'nufft', f.nufft);
Ge = Gmri(ksp, ig.mask, 'exact', 1, 'n_shift', N/2, 'fov', f.fov, ...
	'basis', {f.basis});

ftrue=smoothn(2*pi*fmap(:,:,islice),2,'Robust'); % obtained from separate B0 scan

%finit = mri_field_map_reg1(xunc, [f.te2
%f.te1],'winit',fmap,'mask',mask);%%% this function can be used if there is
%time shift between UP and DOWN scans to have self consistent B0 from the
%same EPI scans
 ztrue = 1i*ftrue;


    
% finit = mri_field_map_reg1(xunc_concat, [f.te2 f.te1],'winit',finit,'mask',mask);

%load('/cs/research/vision/home0/green/musman/thindrives/d/UCL_Matlab_data/patient12/SENSE/kspace_up')
%load('/cs/research/vision/home0/green/musman/thindrives/d/UCL_Matlab_data/patient12/SENSE/kspace_down')
%load('/cs/research/vision/home0/green/musman/thindrives/d/b0_joint_p13')

%%% initialize regularization parameters with default values, will be
%%% optimized later
b1=0.04;
R1 = Robject(mask, 'beta', b1, 'potential', 'quad', 'order', 2);
b2=0.001;
R2 = Robject(mask, 'beta', b2, 'potential', 'quad', 'order', 2);

%% default number of outer iterations of CG
   f.niter = 20;


%correction for b-value of 0 with initial B0
b_value=1;
b0_joint=ztrue; %% initial B0 from map (B0 scan)
avg=1;dir=1;Sk11=double(squeeze(kspace_up(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
Sk22=double(squeeze(kspace_down(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
y=[];
y=[Sk11(:);Sk22(:)];Gmp2 = feval(Gm.arg.new_zmap,Gm,ti,b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc2{ic} = Gmp2 * diag_sp(tmp(ig.mask)); % cascade
end
Gb2 = block_fatrix(Gc2, 'type', 'col'); % [G1; G2; ... ]
xr = qpwls_pcg1(0*zeros(row,col), Gb2, f.wi, y, R1.C, 'niter', 50);
xunc = ig.embed(xr);
rec_corr_b0_b0initial=xunc;
save rec_corr_b0_b0initial rec_corr_b0_b0initial;

%%%% No B0 correction for b-value of 0
b0_joint=zeros(row,col);%% initial B0 zero
avg=1;dir=1;Sk11=double(squeeze(kspace_up(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
Sk22=double(squeeze(kspace_down(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
y=[];
y=[Sk11(:);Sk22(:)];Gmp2 = feval(Gm.arg.new_zmap,Gm,ti,b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc2{ic} = Gmp2 * diag_sp(tmp(ig.mask)); % cascade
end
Gb2 = block_fatrix(Gc2, 'type', 'col'); % [G1; G2; ... ]
xr = qpwls_pcg1(0*zeros(row,col), Gb2, f.wi, y, R1.C, 'niter', 50);
xunc = ig.embed(xr);
rec_nocorr_b0=xunc;
save rec_nocorr_b0 rec_nocorr_b0
%%%end


%joint image and B0 estimation for b-value of 0 
%%% No B0 initialization
b_value=1;
b0_joint=zeros(row,col);%% initial B0 zero
b_value=1;
avg=1;dir=1;Sk11=double(squeeze(kspace_up(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
Sk22=double(squeeze(kspace_down(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
y=[];
y=[Sk11(:);Sk22(:)];Gmp2 = feval(Gm.arg.new_zmap,Gm,ti,b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc2{ic} = Gmp2 * diag_sp(tmp(ig.mask)); % cascade
end
Gb2 = block_fatrix(Gc2, 'type', 'col'); % [G1; G2; ... ]
xr = qpwls_pcg1(0*zeros(row,col), Gb2, f.wi, y, R1.C, 'niter', 50);
xunc = ig.embed(xr);

zmap1 = b0_joint; %zeros(nx,ny); %
xcg1 = xunc; %zero
parameter_setting=0;
%%% number of regularization values for which FWHM of PSF is to be tested
%%% the highest reg value for which criteria FWHM<1.01 pixels is satisfied is set
%%% to be the optimized value
values=0.3:-0.01:0.01;
for i=1:20 %% outer CG iterations
printm 'xcg1 and zmap1 iterative'
Gm = feval(Gm.arg.new_zmap, Gm, ti, zmap1, f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc{ic} = Gm * diag_sp(tmp(ig.mask)); % cascade
end
Gb = block_fatrix(Gc, 'type', 'col'); % [G1; G2; ... ]
cpu tic
f.niter = 50; %% CG iterations of inner subproblem image estimation
vec=[];
if parameter_setting==0;
for ii=1:length(values);b1 = values(ii);%2^-22
R1 = Robject(mask, 'beta', b1, 'potential', 'quad', 'order', 2);
[psf,var,fwhm]=qpwls_psf(Gb, R1, 1, mask, 1, 'chat', 0);
vec=[vec;fwhm];
if(fwhm<1.01);%%% if condition FWHM<1.01 pixels is satisfied, set to the optimal value for b1
break;
end;
end;
b1=values(ii);
%%% use that b1 for regularization object R1 and in CG subproblem
R1 = Robject(mask, 'beta', b1, 'potential', 'quad', 'order', 2);
vec=[];
end
xcg1 = qpwls_pcg1(xcg1(mask), Gb, 1, y(:), R1.C, 'niter', f.niter);
xcg1 = embed(xcg1(:,end), mask);
F = diag_sp(xcg1(mask));
Gz = T*Gb*F;%% operator for B0 estimation
vec=[];
if parameter_setting==0;
    
    %%% now test for B0 estimation the regularization values b2.
for ii=1:length(values);b2 = values(ii);%2^-22
R2 = Robject(mask, 'beta', b2, 'potential', 'quad', 'order', 2);
[psf,var,fwhm]=qpwls_psf(Gz, R2, 1, mask, 1, 'chat', 0);
vec=[vec;fwhm];
if(fwhm<1.2);%% if condition FWHM<1.2 pixels is satisfied, set to the optimal value for b2
break;
end;
end;
b2=values(ii);
R2 = Robject(mask, 'beta', b2, 'potential', 'quad', 'order', 2);
vec=[];
parameter_setting=1;
end
yz = y(:) - Gb*xcg1(mask) + Gz*zmap1(mask);
f.niter = 50; %%% nuber of CG iterations for inner B0 estimation problem
zmap1 = qpwls_pcg1(zmap1(mask), Gz, 1, yz(:), R2.C, 'niter', f.niter);
zmap1 = embed(zmap1(:,end), mask);
end; %%% outer loop CG
rec_corr_b0=xcg1;% final corrected image
B0_corr_b0=zmap1;% final corrected B0
save rec_corr_b0 rec_corr_b0 %%% corrected image at b-value=0
save B0_corr_b0 B0_corr_b0  %%% corrected B0 map at b-value=0

%%% USE corrected B0 map for reconstruction at b-value >0 
b0_joint=zmap1;%%% final B0 map estimated at b-value of 0
%%% use final estimated B0 map for b-values>0
%%% for each avg and for each diffusion direction
b_value=2;for avg=1:size(kspace_up,12);for dir=1:size(kspace_up,11);Sk11=double(squeeze(kspace_up(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
Sk22=double(squeeze(kspace_down(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
y=[];
temp1=Sk11;
temp2=Sk22;
Gmp1 = feval(Gu.arg.new_zmap,Gu,ti(1:end/2),b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc1{ic} = Gmp1 * diag_sp(tmp(ig.mask)); % cascade
end
Gb1 = block_fatrix(Gc1, 'type', 'col'); xr = qpwls_pcg1(0*zeros(row,col), Gb1, f.wi, Sk11(:), R1.C, 'niter', 30);
xunc1 = ig.embed(xr);
Gmp1 = feval(Gu.arg.new_zmap,Gu,ti(1+end/2:end),b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc1{ic} = Gmp1 * diag_sp(tmp(ig.mask)); % cascade
end
Gb1 = block_fatrix(Gc1, 'type', 'col'); xr = qpwls_pcg1(0*zeros(row,col), Gb1, f.wi, Sk22(:), R1.C, 'niter', 30);
xunc2 = ig.embed(xr);
if strcmp(reference,'DOWN')

    angle1=angle(xunc1.*conj(xunc2));im11=fftshift(ifft2(ifftshift(Sk11)));sk11=im11.*exp(-1i*angle1);
Sk11=fftshift(fft2(ifftshift(sk11)));y=[Sk11(:);Sk22(:)];
else;
    angle1=angle(xunc2.*conj(xunc1));im22=fftshift(ifft2(ifftshift(Sk22)));sk22=im22.*exp(-1i*angle1);
Sk22=fftshift(fft2(ifftshift(sk22)));y=[Sk11(:);Sk22(:)];
end
Gmp2 = feval(Gm.arg.new_zmap,Gm,ti,b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc2{ic} = Gmp2 * diag_sp(tmp(ig.mask)); % cascade
end
Gb2 = block_fatrix(Gc2, 'type', 'col'); % [G1; G2; ... ]
xr = qpwls_pcg1(0*zeros(row,col), Gb2, f.wi, y, R1.C, 'niter', 30);
xunc = ig.embed(xr);rec_corr(:,:,dir,avg)=xunc;end;end;
save rec_corr rec_corr; %%% final B0 corrected reconstruction for b-value>0


%%% Results for b-value>0 without B0 correction
b_value=2;for avg=1:size(kspace_up,12);for dir=1:size(kspace_up,11);Sk11=double(squeeze(kspace_up(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
Sk22=double(squeeze(kspace_down(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
y=[];
temp1=Sk11;
temp2=Sk22;
y=[Sk11(:);Sk22(:)];Gmp2 = feval(Gm.arg.new_zmap,Gm,ti,0.*b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc2{ic} = Gmp2 * diag_sp(tmp(ig.mask)); % cascade
end
Gb2 = block_fatrix(Gc2, 'type', 'col'); % [G1; G2; ... ]
xr = qpwls_pcg1(0*zeros(row,col), Gb2, f.wi, y, R1.C, 'niter', 30);
xunc = ig.embed(xr);rec_nocorr(:,:,dir,avg)=xunc;end;end;
save rec_nocorr rec_nocorr;%% results uncorrected reconstruction b-value>0


%%% Results for b-value>0 with initial B0 correction
b0_joint=ztrue;
b_value=2;for avg=1:size(kspace_up,12);for dir=1:size(kspace_up,11);Sk11=double(squeeze(kspace_up(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
Sk22=double(squeeze(kspace_down(:,:,1,1,1,1,1,islice,1,b_value,dir,avg)));
y=[];
temp1=Sk11;
temp2=Sk22;
Gmp1 = feval(Gu.arg.new_zmap,Gu,ti(1:end/2),b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc1{ic} = Gmp1 * diag_sp(tmp(ig.mask)); % cascade
end
Gb1 = block_fatrix(Gc1, 'type', 'col'); xr= qpwls_pcg1(0*zeros(row,col), Gb1, f.wi, Sk11(:), R1.C, 'niter', 30);
xunc1 = ig.embed(xr);
Gmp1 = feval(Gu.arg.new_zmap,Gu,ti(1+end/2:end),b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc1{ic} = Gmp1 * diag_sp(tmp(ig.mask)); % cascade
end
Gb1 = block_fatrix(Gc1, 'type', 'col'); xr = qpwls_pcg1(0*zeros(row,col), Gb1, f.wi, Sk22(:), R1.C, 'niter', 30);
xunc2 = ig.embed(xr);

if strcmp(reference,'DOWN')

    angle1=angle(xunc1.*conj(xunc2));im11=fftshift(ifft2(ifftshift(Sk11)));sk11=im11.*exp(-1i*angle1);
Sk11=fftshift(fft2(ifftshift(sk11)));y=[Sk11(:);Sk22(:)];
else;
    angle1=angle(xunc2.*conj(xunc1));im22=fftshift(ifft2(ifftshift(Sk22)));sk22=im22.*exp(-1i*angle1);
Sk22=fftshift(fft2(ifftshift(sk22)));y=[Sk11(:);Sk22(:)];
end

Gmp2 = feval(Gm.arg.new_zmap,Gm,ti,b0_joint,f.tseg);
for ic=1:f.ncoil
tmp = f.smap(:,:,ic);
Gc2{ic} = Gmp2 * diag_sp(tmp(ig.mask)); % cascade
end
Gb2 = block_fatrix(Gc2, 'type', 'col'); % [G1; G2; ... ]
xr = qpwls_pcg1(0*zeros(row,col), Gb2, f.wi, y, R1.C, 'niter', 30);
xunc = ig.embed(xr);rec_init(:,:,dir,avg)=xunc;end;end;
save rec_init rec_init; %% results for correction using single B0 map for b-value>0

