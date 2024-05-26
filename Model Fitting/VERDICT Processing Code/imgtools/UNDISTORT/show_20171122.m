function show_20171122
% Code to demo reading of raw data.
%
%
% David Atkinson  D.Atkinson@ucl.ac.uk

disp(['Select raw data folder'])
pnr = pref_uigetdir('show_20171122','pnr') ;

% Example files:
fnT2 = 'ob_22112017_1031211_10_2_t2w_tse_ax_clearV4.raw' ;

%%fnDWI = 'ob_22112017_1041511_18_2_wip_dwi_b1_500_1000_pea_minwfs_clearV4.raw' ;
fnDWI = 'ob_22112017_1040503_17_2_wip_dwi_b1_500_1000_pea_maxwfs_clearV4.raw' ;

fnB0 = 'ob_22112017_1035324_14_2_wip_b0map_de_3d_freqoff0_fullprep_f0detauto_clearV4.raw' ;
deltaTE = 2.3 ;

fnepi1 = 'ob_22112017_1038572_16_2_wip_dwi_b1_500_1000_pea_freqoff0_autoprep_f0detauto_intf0__clearV4.raw' ;
fnepi2 = 'ob_22112017_1040503_17_2_wip_dwi_b1_500_1000_pea_maxwfs_clearV4.raw' ;

% Read a T2W image file:
opts = {'verbose',true,'correct_nus',false,'EPIphasecorrection', false, 'MPS_only', true} ;
% The MPS_only means that the rotations,flips and chopping to get into
% radiological presentation are not applied.

[kT2, infoT2] = main_loadLABRAW(fullfile(pnr,fnT2), opts{:});
[imT2, geomT2 ] = explore_recon(kT2, infoT2,'meas',1,'extra',1,'row',1,'MPS_only', true) ;    
disp([fnT2,' IPP ',num2str(geomT2(1).IPP)])
eshow(imT2,'Name',['T2: ',fnT2])
 

% Read B0 file
% % [kB0, infoB0] = main_loadLABRAW(fullfile(pnr,fnB0), opts{:});
% % [imB0, geomB0 ] = explore_recon(kB0, infoB0,'MPS_only',true) ;
% % 
% % coil = 4 % just one coil elememt processed here
% % p2c = imB0(:,:,:,coil,1,1,1,1,1,2) ;
% % 
% % p1c = imB0(:,:,:,coil,1,1,1,1,1,1) ;
% % 
% % pdiff = angle(p2c./p1c) ;
% % 
% % Hz = (pdiff)/(2*pi*deltaTE*1e-3);
% % 
% % eshow(Hz,'Name',['B0 map: ',fnB0])


% Read Diffusion EPI data

epi_opts = {'verbose',true,'correct_nus',true,'EPIphasecorrection', true, 'MPS_only', true} ;
% [kDWI, infoDWI] = main_loadLABRAW(fullfile(pnr,fnDWI),epi_opts{:});
% [imDWI, geomDWI ] = explore_recon(kDWI, infoDWI,'meas',1,'extra',1,'row',1,'MPS_only', true) ;
% 
% eshow(imDWI,'Name',['Diffusion: ', fnDWI] )

[kepi1, infoepi1] = main_loadLABRAW(fullfile(pnr,fnepi1),epi_opts{:});
[imepi1, geomepi1 ] = explore_recon(kepi1, infoepi1,'meas',1,'extra',1,'row',1,'MPS_only', true) ;

[kepi2, infoepi2] = main_loadLABRAW(fullfile(pnr,fnepi2),epi_opts{:});
[imepi2, geomepi2 ] = explore_recon(kepi2, infoepi2,'meas',1,'extra',1,'row',1,'MPS_only', true) ;

% Commented code below puts two EPIs in frame of T2 but needs rotation due to
% rotation of PE in T2 wrt EPIs

% % % rotate T2
% % [imT2r, geomT2r ] = dinplanet( imT2, geomT2, 'rot', 1 ) ;
% % 
% % d1 = dreslice(imepi1, geomepi1, geomT2r) ;
% % d2 = dreslice(imepi2, geomepi2, geomT2r) ;

% Compute d2 at geometry of d1
d2at1 = dreslice(imepi2, geomepi2, geomepi1) ;

figure
coil = 4 ;
imshowpair(abs(imepi1(:,:,1,coil)), abs(d2at1(:,:,1,coil)))









