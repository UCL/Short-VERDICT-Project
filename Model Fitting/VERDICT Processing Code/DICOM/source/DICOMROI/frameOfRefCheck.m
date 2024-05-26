function uniqueFrameOfReferenceUID = frameOfRefCheck(rtinfo, rtContours)
%
% uniqueFrameOfReferenceUID = frameOfRefCheck(rtinfo)
% uniqueFrameOfReferenceUID = frameOfRefCheck(rtinfo, rtContours)
%
% If 'all' the FrameOfReferenceUIDs in the RTstruct file agree, this is
% returned in uniqueFrameOfReferenceUID. Otherwhise, this is returned as
% ''.
%
% See also indexRT display_rtstruct sviewer

if nargin < 2
    rtContours = dicomContours(rtinfo) ;
end

strucSetRefFORUID = rtinfo.ReferencedFrameOfReferenceSequence.Item_1.FrameOfReferenceUID ;

uniqueFrameOfReferenceUID = strucSetRefFORUID ;

ROIs  = rtContours.ROIs ;
nROIs = size(ROIs,1) ;  % number of ROIS (each ROI can have multiple contours)

for iROI = 1:nROIs
    ROIItemStr = sprintf('Item_%d',iROI) ;

    foruid = rtinfo.StructureSetROISequence.(ROIItemStr).ReferencedFrameOfReferenceUID ;
    if ~strcmp(foruid,strucSetRefFORUID)
        warning('MATLAB:frameOfRefCheck:FoRefUIDmismatch','Frame of Reference UIDs not all same.')
        uniqueFrameOfReferenceUID = '' ;
    end
end

end