function RTcont = replaceRefInfo(refInfo, RTInfo)
% REPLACEREFINFO Replaces the DICOM info from Reference file
% Creates RTcont from Reference file and adds any contours
% that are in RTInfo
%
% RTcont = replaceRefInfo(refInfo, RTInfo)
%
% RTcont is created using refInfo, to capture PatientName etc, and then all
% contours that are in RTInfo are added to RTcont.
% The output RTcont can then be converted to an info structure and used to
% write a DICOM.
%
% Example
%
%   RTcont = replaceRefInfo(refInfo, RTInfo)
%   RTInfo = convertToInfo(RTcont) ;
%   dicomwrite(...)
%
% Copyright 2021, David Atkinson
%
% See also dicomRTStruct dicomContours addRefInfo ref2RT

% class dicomRTStruct is a modification of MATLAB's dicomContour
% The output RTcont has public ROIs and private information about the
% reference DICOM info
RTcont  = dicomRTStruct(refInfo) ; % New RTcont generated from refInfo

% dicomContours is MATLAB code that expects input to be from an RTStruct
contours = dicomContours(RTInfo) ;

ROIs = contours.ROIs ; % Table of contours that are in RTinfo

nroi = size(ROIs,1) ;

for iroi = 1:nroi
    nm = ROIs.Name(iroi) ;
    gt = ROIs.GeometricType(iroi) ;
    ct = ROIs.ContourData(iroi) ;

    if isfield(ROIs,'Color')
        colarg = {ROIs.Color(iroi) };
    else
        colarg = {} ;
    end
    RTcont = addContour(RTcont,ROIs.Number(iroi), nm{1}, ...
        ct{:}, gt{1},colarg{:}) ;
end