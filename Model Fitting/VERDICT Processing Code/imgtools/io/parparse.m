function dinfo = parparse(parfn)
% PARPARSE Parse Philips PAR files
%  dinfo = parparse
%  dinfo = parparse(parfn) 
%
%  Returns dinfo empty if there is an error parsing PAR file.
%
% Intended to be used with d2mat. Mirrors dparse for DICOM files and
% xmlparse for XMLpar files
%
% Copyright, 2019, David Atkinson
% $Id: parparse.m 340 2012-06-17 21:23:08Z ucacdat $
%
% See also DPARSE D2MAT XMLPARSE DMFPARSE

% Still needs computation of ImageOrientationPatient etc

if nargin==0 || ~exist(parfn,'file')
  global PARFN   % ensures name persist across calls to this 
                % function (for ease of use and testing, not essential)
                        
  [fn,pn] = uigetfile({'*.par', '*.PAR'}, 'Select Philips PAR file.', PARFN) ;
  if isequal(fn, 0) ; dinfo = [] ; return; end ;
  PARFN = fullfile(pn,fn) ;
 
  parfn = PARFN ;
end

disp(['Parsing par file: ',parfn])


lab = parreadall(parfn) ;

nim = length(lab.par) ;

% pre-allocate space

%!! add SliceOrientation, offcentre and angulation
dinfo(1,nim) = struct('sl',[],'TemporalPositionIdentifier',[],...
    'RescaleSlope',[],'RescaleIntercept',[],...
    'Private_2005_100e',[], 'PixelSpacing', [], ...
    'Width',[],'Height',[],'DiffusionBValue',[],...
    'SliceOrientation',[], 'DiffusionGradientOrientation', [], ...
    'DiffusionCS', [], 'DiffGradOrientIdentifier', [], ...
    'SeriesNumber',[],'ProtocolName',[],...
    'SliceThickness',[], ...
    'RecOffsetBytes',[],'RecFileSize',[],'RecFileName',[]) ;


seriesno = lab.AcqNr*100 + lab.RecNr ;
disp([' Computing series no. from (100.acq + rec) to be: ',num2str(seriesno)])


for iim = 1:nim
    dinfo(iim).sl = lab.par(iim).loca ;
    dinfo(iim).TemporalPositionIdentifier = lab.par(iim).dyn ;
    dinfo(iim).RescaleSlope = lab.par(iim).RescaleSlope ;
    dinfo(iim).RescaleIntercept = lab.par(iim).RescaleIntercept ;
    dinfo(iim).Private_2005_100e = lab.par(iim).ScaleSlope ;
    dinfo(iim).PixelSpacing = lab.par(iim).PixelSpacing_hw ;
    dinfo(iim).SliceThickness = lab.par(iim).SliceThickness ;
    dinfo(iim).Width = lab.par(iim).sz(2) ;
    dinfo(iim).Height = lab.par(iim).sz(1) ;
    dinfo(iim).ImageAngulation_lph = lab.par(iim).ImageAngulation_lph ;
    dinfo(iim).Offcentre_lph = lab.par(iim).Offcentre_lph ;
    dinfo(iim).DiffusionBValue = lab.par(iim).DiffusionBFactor ;
    dinfo(iim).SliceOrientation = lab.par(iim).SliceOrientation ;
    dinfo(iim).DiffusionGradientOrientation = lab.par(iim).DiffusionGradientOrientation ;
    dinfo(iim).DiffGradOrientIdentifier = lab.par(iim).DiffGradOrientIdentifier ;
    dinfo(iim).SeriesNumber = seriesno ;
    dinfo(iim).ProtocolName = lab.ProtocolName ;
    dinfo(iim).RecOffsetBytes = lab.par(iim).RecOffsetBytes ;
    dinfo(iim).RecFileSize = lab. RecFileSize ;
    dinfo(iim).RecFileName = lab.RecFileName ;
end

% Generate DICOM geometries
warning(['Creating DICOM geometries. This code is not fully tested'])

for iim = 1:nim
    ori = dinfo(iim).SliceOrientation ;
    offc = dinfo(iim).Offcentre_lph  ;
    ang = dinfo(iim).ImageAngulation_lph ;
    PS_HW = dinfo(iim).PixelSpacing ;
    Width = dinfo(iim).Width ;
    Height = dinfo(iim).Height ;
    
    [iop, ipp] = dgeom(offc, ang, ori, PS_HW , Width, Height) ;
    
    dinfo(iim).ImageOrientationPatient = iop ;
    dinfo(iim).ImagePositionPatient = ipp ;
end

THRESH = 0.015 ;
dinfo = data_process(dinfo, THRESH) ;

end




