function [MRSeriesWaterFatShift_scaled, wfs_hzpp_scaled, ...
    fat_shift_dir_lph, dout, shift_hzpmm  ] = dwfs(Dtype, file_descrip )
%DWFS DICOM Water-fat-shift information
%   User selects either Enhanced MR and corresponding RAW (XX) file at
%   once, or, representative Single Frame and the XX file
%   Philips format only with private fields present
%
% Asumes only one stack and parallel slices.
%
% David Atkinson  D.Atkinson@ucl.ac.uk  
%
% See also dsfsort XXread

PPM_WFS = 3.4 ; % WFS in ppm (value used by Philips)

% From enum for Philips AWGEOM_DIR_ENUM  R,L,A,P,F,H (0 based enumeration)
% typedef enum
% {			/* Definition of directions in patient frame	*/
%     AWGEOM_DIR_MIN = -1,
%     AWGEOM_DIR_R,
%     AWGEOM_DIR_L,
%     AWGEOM_DIR_A,
%     AWGEOM_DIR_P,
%     AWGEOM_DIR_F,
%     AWGEOM_DIR_H,
%     AWGEOM_DIR_MAX
% } AWGEOM_DIR_ENUM;

awgeom_dir_enum = {...
    [-1 0 0], [1 0 0], ...
    [0 -1 0], [0 1 0], ...
    [0 0 -1], [0 0 1] } ;

if nargin > 1
    fstr = ['For ',file_descrip,' select '] ;
else
    fstr=['Select '];
end

switch Dtype
    case 'EMR' % Enhanced MR
        disp([fstr,'Enhanced DICOM file (EMR) and corresponding XX (RAW) file'])
        
        fns = getdfiles(dselector, 'EMR and corresponding XX file') ;
        
        if length(fns)~=2
            warning('Need to select two files')
        end
        
        dinf1 = dicominfo(fns{1}) ;
        dinf2 = dicominfo(fns{2}) ;
        
        if dinf1.SeriesNumber ~= dinf2.SeriesNumber
            warning(['Selected files are not from the same series'])
        end
        
        if ~strcmp(dinf1.Manufacturer,'Philips Medical Systems')
            warning(['Only Philips files supported.'])
        end
        
        switch dinf1.SOPClassUID
            case '1.2.840.10008.5.1.4.1.1.4.1'
                demr = dinf1 ;
                dxx  = dinf2 ;
            case '1.2.840.10008.5.1.4.1.1.66'
                dxx  = dinf1 ;
                demr = dinf2 ;
            otherwise
                warning(['Expected EMR and RAW DICOMs'])
        end
        
        clear dinf1 dinf2
        
        
        % WFS properties
        MRSeriesWaterFatShift = demr.Private_2001_1022 ;
        MRSeriesImagingFrequency = demr.Private_2001_1083 ;
        PixelBandwidth = demr.PixelBandwidth ;
        PixelBW = demr.SharedFunctionalGroupsSequence.Item_1.MRImagingModifierSequence.Item_1.PixelBandwidth ;
        pes = demr.SharedFunctionalGroupsSequence.Item_1.MRFOVGeometrySequence.Item_1.InPlanePhaseEncodingDirection ;
        iop = demr.PerFrameFunctionalGroupsSequence.Item_1.PlaneOrientationSequence.Item_1.ImageOrientationPatient ;
        
        xxinfo = XXread(dxx.Filename) ;
        dout = demr ;
    case 'Single'
        disp([fstr,'representative Single Frame file'])
        
        sf_fn = pref_uigetfile('dwfs','sfdicom') ;
        dsf = dicominfo(sf_fn) ;
        
        MRSeriesWaterFatShift = dsf.Private_2001_1022 ;
        MRSeriesImagingFrequency = dsf.Private_2001_1083 ;
        PixelBandwidth = dsf.PixelBandwidth ;
        
        % uncertain if these are just to a decimal point or not
        PixelBW = PixelBandwidth ;
        pes = dsf.InPlanePhaseEncodingDirection ;
        iop = dsf.ImageOrientationPatient ;
        
        disp(['Select XX file corresponding to ',sf_fn,'.'])
        xxinfo = XXread ;
        dout = dsf ;
        
    otherwise
        error(['Unknown Dtype: ',Dtype])
end


wfs_fromBW = PPM_WFS * MRSeriesImagingFrequency / PixelBW ;

% For EPI, these will not be the same.
disp(['WFS: ',num2str(MRSeriesWaterFatShift),', (',num2str(wfs_fromBW),' from PixelBandwidth)'])



fat_shift_dir_lph = awgeom_dir_enum{xxinfo.EX_GEO_cur_stack_fat_shift_dir + 1 } ;

% below is shift in Hz per pixel (derived from WFS, but not fat-specific)
wfs_hzpp = PPM_WFS * MRSeriesImagingFrequency / MRSeriesWaterFatShift ;

disp(['WFS direction: ',num2str(fat_shift_dir_lph),' (LPH)', ...
    ' HZ per pixel: ',num2str(wfs_hzpp)])

% Actual and recon voxel sizes (WFS is probably in actual voxel size, hence
% need to scale for recon voxel size in DICOM).

avs = xxinfo.IF_meas_voxel_size ; % from info tab
rvs = xxinfo.IF_recon_voxel_size ;

av =  textscan(avs{1},'%f / %f / %f')  ; % convert from string to cell
rv =  textscan(rvs{1},'%f / %f / %f')  ;

disp(['Aq voxels: ',av])
disp(['Recon voxels: ',rv])

% Is WFS in readout or PE direction of image?
% Assumes all slices parallel.

switch pes
    case {'COLUMN', 'COL' }
        pe_dir = iop(4:6) ;
        ro_dir = iop(1:3) ;
    case 'ROW'
        pe_dir = iop(1:3) ;
        ro_dir = iop(4:6) ;
    otherwise
        error(['Unknown PE direction'])
end

if abs(dot(pe_dir,fat_shift_dir_lph)) > abs(dot(ro_dir,fat_shift_dir_lph))
    % fat shift is in PE direction
    disp(['Fat shift is in phase encode direc (expected for EPI)'])
    wfs_scale = av{2}/rv{2} ;
    rec = rv{2} ;
else
    disp(['Fat shift is in readout direc'])
    wfs_scale = av{1}/rv{1} ;
    rec = rv{1} ;
end

MRSeriesWaterFatShift_scaled = MRSeriesWaterFatShift * wfs_scale ;
% Units of wfs are pixels, not Hz/pixel
wfs_hzpp_scaled = wfs_hzpp / wfs_scale ;
shift_hzpmm = wfs_hzpp_scaled * rec ;  % units are mm


disp(['After scaling for aq/recon voxel sizes, WFS: ',num2str(MRSeriesWaterFatShift_scaled), ...
    ', BWpp: ',num2str(wfs_hzpp_scaled)])


end

