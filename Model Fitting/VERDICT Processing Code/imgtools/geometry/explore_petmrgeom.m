function [twix_obj, dset, dinfo] = explore_petmrgeom

data_folder = pref_uigetdir('petmrgeom','folder') ;
dcm_hfs_folder = fullfile(data_folder, 'ccp_calib_HFS','Biograph_mMR','MR',...
    '2018-08-22','174706_109000','t2_tse_sag_3') ;

mr_raw_folder = 'CCP_CALIB_MR' ;

% t2_tse_sag
ds.mr_stem = 'meas_MID00749_FID151779_t2_tse_sag' ;  % HFS
% meas_MID00828_FID151857_t2_tse_sag.dat  % FFS

% MR raw data and hdf5 file.
mr_dat_fn = fullfile(data_folder,mr_raw_folder,[ds.mr_stem, '.dat']) ;
mr_h5_fn = fullfile(data_folder,mr_raw_folder,[ds.mr_stem, '.h5']) ;


% DICOM
dr = dir(dcm_hfs_folder) ;
dinfo = datparse(fullfile(dr(3).folder, dr(3).name)) ;

% Gadgetron recon
gadg_fn = fullfile(data_folder, mr_raw_folder,[ds.mr_stem, '_recon.h5']) ;
disp(['TEMP file name'])
gadg_fn = fullfile(data_folder, mr_raw_folder,'reconfs_749.h5') ;

[fn, dsn] = explore_h5(gadg_fn);
gdata = h5read(fn,dsn{2}) ;
hdr = h5read(fn,dsn{3}) ;

%  get gadgetron fs.h5 file
%  convert to complex

sl = 10 ;
zv = gdata.real + 1i*gdata.imag ;
z = squeeze(zv(:,:,1,1,sl)) ;

%  find positions, assume centre of slice
nr = size(z,1) ;
nc = size(z,2) ;

% row & col
iopr =  hdr.read_dir(:,sl) ;
iopc = hdr.phase_dir(:,sl) ;

cpos = hdr.position(:,sl) ; % centre pos?

% need pixel size
tlc = cpos + (-iopr.* floor((nr+1)/2)) - (iopc.*floor((nc+1)/2) ) 



%  find orientations
%  compute tlc IPP
%  call inplanet to get radiological.
%  
twix_obj = mapVBVD(mr_dat_fn) ;

dset = ismrmrd.Dataset(mr_h5_fn, 'dataset');
D = dset.readAcquisition(); 
    
% We want to get out the PixelSpacing for aftr recon, orientations, bed
% shifts. Compare with ISMRMRD

twix_obj.hdr.Dicom.lSBCSOriginPositionX
twix_obj.hdr.Dicom.lSBCSOriginPositionY
twix_obj.hdr.Dicom.lSBCSOriginPositionZ

twix_obj.hdr.Meas.SBCSOriginPositionX
twix_obj.hdr.Meas.SBCSOriginPositionY
twix_obj.hdr.Meas.SBCSOriginPositionZ

twix_obj.hdr.Meas.TableHomeZPosition

twix_obj.hdr.Meas.lOffsetX
twix_obj.hdr.Meas.lOffsetY
twix_obj.hdr.Meas.lOffsetZ

twix_obj.hdr.Meas.lSBCSOriginPositionX
twix_obj.hdr.Meas.lSBCSOriginPositionY
twix_obj.hdr.Meas.lSBCSOriginPositionZ


% Do a MATLAB ISMRMRD recon, write out DICOM and check, then compare to
% Gadgetron

% From ISMRMRD k-space file. Use matrix size and fov of encodedSpace -
% appears to be small rounding error. Assume RO centre is at centre. So,
% fill k-space, remeber that adjusting k-space does not change FOV.
% Look at explore ...
% 


