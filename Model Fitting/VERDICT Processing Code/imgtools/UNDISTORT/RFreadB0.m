function RFreadB0
% RFreadB0
%
% ReconFrame free reading of B0 map data.
%
% Uses Philips MATLB code.
%
%

nonEPI = {'correct_nus',false,'EPIphasecorrection', false} ;

rawfn = pref_uigetfile('UNDISTORT','RFreadB0') ;
[rawpn,~, rawext] = fileparts(rawfn) ;

[bdata, binfo] = main_loadLABRAW(rawfn,'verbose',true, nonEPI{:});
[~,sense_stem, ~] = fileparts(binfo.sin.senseref) ;
sensefn = fullfile(rawpn, [sense_stem,rawext]) ;
if ~exist(sensefn,'file')
    warning(['SENSE raw file doesnt exist: ',sensefn])
end

[sdata, sinfo] = main_loadLABRAW(sensefn,'verbose',true, nonEPI{:} ) ;
[~,coilsurvey_stem,~] = fileparts(sinfo.sin.coilsurvey) ;
coilsurveyfn = fullfile(rawpn, [coilsurvey_stem, rawext]) ;
if ~exist(coilsurveyfn,'file')
    warning(['Coilsurvey raw file doesnt exist: ',coilsurveyfn])
end

[csdata, csinfo] = main_loadLABRAW(coilsurveyfn,'verbose',true, nonEPI{:} ) ;
cs1 = csdata(:,:,:,:,1,1) ; % Has  2 images (QUAD1 and 2?) [96    31    60    32]
%cs2 = csdata(:,:,:,:,1,2) ; % Has 32 'images' for prostate cardiac coil

s1 = sdata(:,:,:,:,1,1) ; % 32 'images' [128    63    48    32]
%s2 = sdata(:,:,:,:,1,2) ; % All zero
    
[  sidata, sgeom  ] = explore_recon(sdata, sinfo,'loc',1);
[ csidata, csgeom ] = explore_recon(csdata, csinfo,'loc',1);

% To resample coils into space of B0, need to figure out B0 geom after
% recon.
% Find dataset with 3D and not SENSE.
% Ensure we can get geoms without going the FFT - need to resampling of
% SENSE.
[ bidata, bgeom ] = explore_recon(bdata, binfo, 'row', 1) ;



disp(['kx,ky,kz, ncoil,E3, Location, Echo, Dynamic, Cardiac, Row, Extra, Meas, Mix'])
size(data)

% Get two images and compare phases

de1 = squeeze(data(:,:,:,:,1,1,1,1,1,1,1,1,1)) ;
de2 = squeeze(data(:,:,:,:,1,1,1,1,1,2,1,1,1)) ;

% [nx ny nz nc]

% Got k-space with PDA and DC corrections (check)
% Now need to assign k-space PE coordinates so that we can account for
% SENSE and partial Fourier. Then do SENSE/partial F recon. 
%
% Need coil sensitivities!
%

