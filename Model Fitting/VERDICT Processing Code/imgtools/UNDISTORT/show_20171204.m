function show_20171204
% Code to demo reading of raw data.
%
%
% David Atkinson  D.Atkinson@ucl.ac.uk

disp(['Select raw data folder'])
pnr = pref_uigetdir('show_20171204','pnr') ;


 fnepi1 = 'ob_04122017_1918024_5_2_wip_dwi_b1_500_1000_pea_minwfs_clearV4.raw' ;
 fnepi2 = 'ob_04122017_1922462_7_2_wip_dwi_b1_500_1000_pep_minwfs_clearV4.raw' ;
%fnepi1 = 'ob_04122017_1948435_17_2_wip_dwi_b1_500_1000_pea_maxwfs_offset0_experiment1_clearV4.raw' ;
%fnepi2 = 'ob_04122017_1950375_19_2_wip_dwi_b1_500_1000_pep_maxwfs_offset0_experiment1_clearV4.raw' ;

% Read Diffusion EPI data

epi_opts = {'verbose',true,'correct_nus',true,'EPIphasecorrection', true, 'MPS_only', true} ;


[kepi1, infoepi1] = main_loadLABRAW(fullfile(pnr,fnepi1),epi_opts{:});
[imepi1, geomepi1 ] = explore_recon(kepi1, infoepi1,'meas',1,'extra',1,'row',1,'MPS_only', true) ;

[kepi2, infoepi2] = main_loadLABRAW(fullfile(pnr,fnepi2),epi_opts{:});
[imepi2, geomepi2 ] = explore_recon(kepi2, infoepi2,'meas',1,'extra',1,'row',1,'MPS_only', true) ;


% Compute d2 at geometry of d1
d2at1 = dreslice(imepi2, geomepi2, geomepi1) ;

figure
coil = 4 ;
imshowpair(abs(imepi1(:,:,1,coil)), abs(d2at1(:,:,1,coil)))









