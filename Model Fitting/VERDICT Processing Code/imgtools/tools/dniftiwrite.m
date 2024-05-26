function dniftiwrite(vol, ffn, geom)
% DNIFTIWRITE Write NIfTI1 file with Sform from DICOM-style geom
%  Currently writes as datatype uint16 (scaled to fit in range).
%  Only handles right-handed data.
%  Code assebles the MATLAB NIfTI1 header
%
% dniftiwrite(vol, ffn, geom)
%
% Copyright 2020, David Atkinson
% D.Atkinson@ucl.ac.uk
%
% See also niftiwrite docom2intrinsic geom_check
%

% Populate MATLAB NIfTI header (not the raw component)
szvol = size(vol) ;
ImageSize = szvol ;
ImageSize(2) = szvol(1) ; ImageSize(1) = szvol(2) ; 


% permute first two dimensions of vol for compatibility with non-MATLAB
% readers (MATLAB writes column-wise)
vol = permute(vol, [2 1 3:ndims(vol)]) ;

% Data Scaling
dtype = 'uint16' ; 
dt_max = intmax(dtype) ;
dt_min = intmin(dtype) ;
dtrange = dt_max - dt_min ;
vmax = max(vol(:)) ;
vmin = min(vol(:)) ;
vrange = vmax-vmin ;
dtrange = cast(dtrange,'like',vrange) ;

if vrange > dtrange
    scl_slope = vrange / dtrange ;
elseif vrange < dtrange
    rslope = floor(dtrange / vrange) ; % 'preserves' integers
    scl_slope = 1/rslope ;
else
    scl_slope = 1 ;
end

scl_inter = vmin ;

% Apply reverse scaling to data and cast to type
vol = (vol-scl_inter)/scl_slope ;
vol = cast(vol, dtype) ;

% In NIfTI the slice spacing is the slice centre separation.
% Force use of right-handed system for simplicity.
slsep = geom_check(geom, 'assertRH', true) ;

ninfo.Version = 'NIfTI1' ;
ninfo.Description = 'dniftiwrite generated' ;
ninfo.ImageSize = ImageSize ;
ninfo.PixelDimensions = [geom(1).PixelSpacing_HW(2)  ...
    geom(1).PixelSpacing_HW(1) slsep ] ;

ninfo.SpaceUnits = 'Millimeter' ;
ninfo.TimeUnits = 'Second' ;
ninfo.SliceCode = 'Unknown' ;
ninfo.FrequencyDimension = 0 ; 
ninfo.PhaseDimension = 0 ; 
ninfo.SpatialDimension = 0 ; 
ninfo.TransformName = 'Sform' ;
ninfo.Qfactor = 1ff ;
ninfo.AdditiveOffset = scl_inter ;
ninfo.MultiplicativeScaling = scl_slope ;
ninfo.Datatype = dtype ;

S = dicom2intrinsic(geom, 'output', 'MATLAB_nifti_sform') ;

ninfo.Transform = affine3d(S) ;

niftiwrite(vol, ffn, ninfo) 
disp(['Written file: ',ffn])


