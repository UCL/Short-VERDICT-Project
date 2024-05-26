function [droi] = ref2RT(refInfo, varargin)
% REF2RT Generate DICOM meta info for RTStruct from Referenced file
%
%  RTMetadata = ref2RT(refInfo, Name, Value, ...)
%
% Name, Value pairs:
%  'RTInstanceNumber', [1]
%  'RTSeriesNumber'   [same as Referenced Image]
%  'StructureSetLabel' ['dicom roi']
%  
% Note this code assigns a new dicomuid for the tag RTSeriesInstanceUID
% which may not be consistent with the above for RTSeriesNumber ?
%
% Called within validateMetadata method of dicomRTStruct
%
% For information on SOPs and what references what:
% See https://www.researchgate.net/profile/Wayne-Newhauser/publication/264241699/figure/fig1/AS:579449819066368@1515163010908/Relationships-between-various-UIDs-used-in-DICOM-RT-treatment-plans_W640.jpg
%
% This code does not allow a range of Frames in Reference Enhanced
% DICOM to be specified, i.e. all Frames possible reference frames
%
% Under development. Testing by reading into Horos or 3D Slicer
% Currently works for single frame DICOM. For multi-frame Enhanced DICOM 
% originals, the contours are not seen in Horos (but they are seen in 3D Slicer). 
% This may be a Horos limitation, note MIMSoft RTStructs also do not show
% up. (But they also dont write out ReferencedFrameNumber).
%
% Future work maybe to bypass addContour so that we write both the contour
% data and any corresponding referencing at the same time. 
% Also need to understand difference between Structure sets and contours
% and where the referenced parts needs updating.
% Overall impression is that for a volume, the referenced image instance
% doesn't matter, so long as FoRUID is correct (at least for Horos).
%
%
% Copyright 2021. David Atkinson
%
% See also dicomRTStruct
%

% Default values
RTInstanceNumber = 1 ;
RTSeriesNumber = refInfo(1).SeriesNumber ; 
StructureSetLabel = 'dicom roi' ;

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    switch varargin{ipv}
        case 'RTInstanceNumber'
            RTInstanceNumber = val ;
        case 'RTSeriesNumber'
            RTSeriesNumber = val ;
        case 'StructureSetLabel'
            StructureSetLabel = val ;

        otherwise
            error(['Unknown Name: ',varargin{ipv}])
    end
end

nRef = length(refInfo) ;

% Reference properties
RefSOPInstanceUID{1} = refInfo(1).SOPInstanceUID ;
RefSOPClassUID = refInfo(1).SOPClassUID ;
SeriesInstanceUID = refInfo(1).SeriesInstanceUID ;
StudyInstanceUID = refInfo(1).StudyInstanceUID ;

% Set FrameOfReferenceUID. Allow only 1 in this code.
% See dmfparse and datparse for reading IOP, IPP etc esp slice centre sep

FoRUID = refInfo(1).FrameOfReferenceUID ;


if isequal(refInfo(1).SOPClassUID, '1.2.840.10008.5.1.4.1.1.4.1' )
    % Enhanced MR

    if nRef > 1 
      error('For Enhancd MR, currently only 1 file can be processeed')
    end

else % Single-Frame Classic
    for iref = 2:nRef

        RefSOPInstanceUID{iref} = refInfo(iref).SOPInstanceUID ;

        if ~isequal(FoRUID, refInfo(iref).FrameOfReferenceUID )
            error('Not all FrameOfReferenceUIDs are the same')
        end

        if ~isequal(RefSOPClassUID, refInfo(iref).SOPClassUID )
            error('Not all SOPClassUIDs are the same')
        end

        if ~isequal(SeriesInstanceUID, refInfo(iref).SeriesInstanceUID )
            error('Not all SeriesInstanceUIDs are the same')
        end

        if ~isequal(StudyInstanceUID, refInfo(iref).StudyInstanceUID )
            error('Not all StudyInstanceUIDs are the same')
        end
    end

end

% Set RTMetadata (here called droi because cut and paste from older code)
RTSOPInstanceUID    = dicomuid;
RTSeriesInstanceUID = dicomuid ;

droi.Format = 'DICOM' ;
droi.FormatVersion = 3 ;
droi.Width = [] ;
droi.Height = [] ;
droi.BitDepth = [] ;
droi.ColorType = [] ;
% FileMetaInformationGroupLength
% FileMetaInformationVersion 
droi.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' ;  % RT Structure Set Storage
droi.MediaStorageSOPInstanceUID = RTSOPInstanceUID ;
droi.TransferSyntaxUID = '1.2.840.10008.1.2.1' ;  % Explicit VR Little Endian
droi.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.3.100.9.4' ; % MATLAB ??
droi.ImplementationVersionName = 'MATLAB dicomroi' ;
droi.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' ;  % RT Structure Set Storage
droi.SOPInstanceUID = RTSOPInstanceUID ;

dorig = refInfo(1) ;
droi.StudyDate = fromorig(dorig,'StudyDate') ;
droi.StudyTime = fromorig(dorig,'StudyTime') ;
droi.AccessionNumber = fromorig(dorig,'Accessionumber') ;
droi.Modality = 'RTSTRUCT' ;
droi.Manufacturer = fromorig(dorig, 'Manufacturer') ;
droi.InstitutionName = fromorig(dorig,'InstitutionName') ;
droi.ReferringPhysicianName = fromorig(dorig, 'ReferringPhysicianName') ;
droi.StationName = fromorig(dorig, 'StationName') ;
droi.SeriesDescription = fromorig(dorig, 'SeriesDescription') ;
droi.ManufacturerModelName = fromorig(dorig, 'ManufacturerModelName') ;
droi.PatientName = fromorig(dorig, 'PatientName') ;
droi.PatientID = fromorig(dorig, 'PatientID') ;
droi.PatientBirthDate = fromorig(dorig, 'PatientBirthDate') ;
droi.PatientSex = fromorig(dorig, 'PatientSex') ;
droi.StudyInstanceUID = StudyInstanceUID ;
droi.SeriesInstanceUID = RTSeriesInstanceUID ;
droi.StudyID = fromorig(dorig, 'StudyID') ;
droi.SeriesNumber = RTSeriesNumber ;
droi.InstanceNumber = RTInstanceNumber ;
droi.OperatorsName = '' ;
droi.StructureSetLabel = StructureSetLabel ;
droi.StructureSetName = '';
droi.StructureSetDate = '' ;
droi.StructureSetTime = '' ;
droi.ReferencedFrameOfReferenceSequence = struct ;  % empty structure

% Set the ReferencedFrameOfReferenceSequence which does not change with
% contours (here only allow contours from same Ref images).
% This is Type 3 and Optional?? But maybe safest to set??

droi.ReferencedFrameOfReferenceSequence.Item_1.FrameOfReferenceUID = FoRUID ;

% Next one is unclear to me - is this the SOPClass of the Study (e.g. MR),
% RTStruct? Seems to be RTStruct in some other examples and a retired one
% in MIM 1.2.840.10008.3.1.2.3.1.
% Here setting to the SOPClassUID of the MR.
droi.ReferencedFrameOfReferenceSequence.Item_1.RTReferencedStudySequence.Item_1.ReferencedSOPClassUID = RefSOPClassUID ; 
droi.ReferencedFrameOfReferenceSequence.Item_1.RTReferencedStudySequence.Item_1.ReferencedSOPInstanceUID = StudyInstanceUID ;
droi.ReferencedFrameOfReferenceSequence.Item_1.RTReferencedStudySequence.Item_1.RTReferencedSeriesSequence.Item_1.SeriesInstanceUID  = SeriesInstanceUID;

% Last Item_ here should be one per image used to draw contours
items = strcat('Item_',string(1:nRef));
for iref = 1: nRef    
    newItem = items(iref); 
    droi.ReferencedFrameOfReferenceSequence.Item_1.RTReferencedStudySequence.Item_1.RTReferencedSeriesSequence.Item_1.ContourImageSequence.(newItem).ReferencedSOPClassUID = RefSOPClassUID ;
    droi.ReferencedFrameOfReferenceSequence.Item_1.RTReferencedStudySequence.Item_1.RTReferencedSeriesSequence.Item_1.ContourImageSequence.(newItem).ReferencedSOPInstanceUID = RefSOPInstanceUID{iref} ;
    % Not setting FrameNumber or Segment
end  


end



function val = fromorig(dorig, tag)
% FROMORIG Gets a DICOM tag from the input original meta info
if isfield(dorig, tag)
    val = dorig.(tag) ;
else
    val = '' ;
end
end