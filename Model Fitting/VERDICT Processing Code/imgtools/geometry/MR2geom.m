function varargout = MR2geom(MR)
% MR2GEOM MRecon object to geom structure
%
% geom = MR2geom      - UI called
% geom = MR2geom(MR)
%        MR2geom(...)  call geomchecker
%
% Slice thickness is taken from MRecon voxel size.
% Slices in geom take the same order as in MR  - may not be RH
%
% David Atkinson D.Atkinson@ucl.ac.uk
%
% See also GEOMCHECKER  MRecon  

if nargin == 0 || ~isa(MR,'MRecon')
    disp('Select a label file for MRecon')
    ffn = pref_uigetfile('MR2geom','labfn') ;
    if ~exist(ffn,'file'), return, end
    MR = MRecon(ffn) ;
end

voxs = MR.Parameter.Scan.RecVoxelSize ;
if abs(voxs(1)-voxs(2)) > 0.01
    error(['Expecting square voxels.'])
end

nsl = MR.Parameter.Scan.ImagesPerStack ;

[xT,~] = MR.Transform([1 1 1], 'REC','RAF') ; % IPP for slice 1
[xT1,~] = MR.Transform([1 2 1], 'REC','RAF') ;
[xT2,~] = MR.Transform([2 1 1], 'REC','RAF') ;
  
IOP(1:3) = (xT1-xT)/voxs(2) ;
IOP(4:6) = (xT2-xT)/voxs(1) ;
    
if abs(norm(IOP(1:3))-1) > 0.001
    warning('IOP(1:3) not normalised.')
end
if abs(norm(IOP(4:6))-1) > 0.001
    warning('IOP(4:6) not normalised.')
end

Height = MR.Parameter.Encoding.XReconRes ;
Width  = MR.Parameter.Encoding.YReconRes ;

if Height ~= Width
    warning(['Setting above assumed before RotateImage.'])
end

PixelSpacing_HW = voxs(1:2) ;
SliceThickness = voxs(3) ;
    
switch MR.Parameter.Scan.ScanMode
    case '3D'
        MRAqType = '3D' ;
    case 'MS'
        MRAqType = '2D' ;
    otherwise
        warning(['Un implemented ScanMode: ',MR.Parameter.Scan.ScanMode])
        MRAqType = '3D' ;
end
  
for isl = 1:nsl
    [xT,~] = MR.Transform([1 1 isl], 'REC','RAF') ;
    geom(isl).IPP = xT.' ;
    geom(isl).IOP = IOP ;
    geom(isl).Height = Height ;
    geom(isl).Width = Width ;
    geom(isl).PixelSpacing_HW = PixelSpacing_HW ;
    geom(isl).SliceThickness = SliceThickness;
    geom(isl).MRAqType = MRAqType ;
    geom(isl).source = 'MR2geom' ;
end
    
if nargout == 0
    [pn,fn, ext] = fileparts(MR.Parameter.Filename.Parameter) ;
    geomchecker('geom', geom, 'msg', [fn ext]) 
else
    varargout{1} = geom ;
end
