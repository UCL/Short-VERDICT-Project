function [rt2Slice, slice2RT, varargout] = indexRT(rtContours, geom, rtinfo)
% indexRT Indices into RTStruct and DICOM volume
%
% Terminology: ROI - a group of contours
%
% [rt2Slice, slice2RT] = indexRT(rtContours, geom)
%
% [rt2Slice, slice2RT] = indexRT(rtContours, geom, rtinfo) checks FoRUIDs
%
% [rt2Slice, slice2RT, rt2RefSOPInstanceUID] = indexRT(rtContours, geom, rtinfo)
%
% rtContours from dicomContour
% geom can originate from multiple series, nSlices = length(geom)
% rtinfo from dicominfo(rtstruct_file)
%
% rt2Slice {iROI} -> contours2slice{iContour} -> [ slices ]
% slice2RT {slice} ->  [n x 2] array where n is number of contours in slice
%  first entry is ROI number, second is contour number within that ROI
%
% rt2RefSOPInstanceUID {iROI} -> contours2refuid{iContour}
%
% See also dicomContours  display_rtstruct

findRefSOPInstanceUID = false ;

if nargin > 2
    strucSetRefFORUID = rtinfo.ReferencedFrameOfReferenceSequence.Item_1.FrameOfReferenceUID ;

    if nargout > 2
        if isfield(rtinfo,'ROIContourSequence')
            if isfield(rtinfo.ROIContourSequence.Item_1,'ContourSequence')
                if isfield(rtinfo.ROIContourSequence.Item_1.ContourSequence.Item_1,'ContourImageSequence')
                    findRefSOPInstanceUID = true;
                end
            end
        end
    end
end

nSlices = length(geom) ; % Note this is spatial slices and not frames

% If its MR 3D, set slicethickness to centre separation, even though not
% strictly correct in sense that slices overlap.
% (Otherwise, get 2 contours per slice).

if strcmp(geom(1).MRAqType,'3D') && nSlices > 1
    normd = cross(geom(1).IOP(1:3),geom(1).IOP(4:6)) ;

    sliceBand = dot(normd, (geom(2).IPP-geom(1).IPP) );
else
    sliceBand = geom(1).SliceThickness ;
end

ROIs  = rtContours.ROIs ;

nROIs = size(ROIs,1) ;  % number of ROIS (each ROI can have multiple contours)
slice2RT = cell([nSlices 1]) ;
rt2Slice = cell([nROIs 1]) ;
rt2RefSOPInstanceUID = cell([nROIs 1]) ;

for iROI = 1:nROIs   % ROI is a group of contours (a row in the table)
    ROIItemStr = sprintf('Item_%d',iROI) ;
    
    if exist('rtinfo','var') 
        foruid = rtinfo.StructureSetROISequence.(ROIItemStr).ReferencedFrameOfReferenceUID ;
        if ~strcmp(foruid,strucSetRefFORUID)
            warning('MATLAB:indexRT:FoRefUIDmismatch','Frame of Reference UIDs not all same.')
        end
    end
    
    contourData = ROIs.ContourData{iROI} ;
    nContours   = size(contourData,1) ;

    contours2slice  = cell([nContours 1]) ; % This is re-created for every ROI
    contours2refuid = cell([nContours 1]) ;

    for iContour = 1:nContours
        if findRefSOPInstanceUID
            contourItemStr = sprintf('Item_%d',iContour) ;

            contours2refuid{iContour} = rtinfo.ROIContourSequence.(ROIItemStr).ContourSequence.(contourItemStr).ContourImageSequence.Item_1.ReferencedSOPInstanceUID ;
        end

        for iSlice = 1:nSlices
            inputCoord3D = contourData{iContour} ;
            [~, distances] = coord2D3D(inputCoord3D,geom(iSlice)) ;

            if max(abs(distances)) <= sliceBand/2 
                slice2RT{iSlice} = cat(1,slice2RT{iSlice}, [iROI iContour]) ;
                contours2slice{iContour} = cat(1,contours2slice{iContour}, iSlice) ;
            end
        end

        if isempty(contours2slice{iContour})
            disp(['No slice found for ROI: ',num2str(iROI),', contour: ',num2str(iContour)])
        end
        if length(contours2slice{iContour}) > 1
            disp([num2str(length(contours2slice{iContour})), ...
                ' slices found for ROI: ',num2str(iROI),', contour: ',num2str(iContour)])
        end

    end

    rt2Slice{iROI} = contours2slice ;
    if findRefSOPInstanceUID
        rt2RefSOPInstanceUID{iROI} = contours2refuid ;
    end

end

if findRefSOPInstanceUID
    varargout{1} = rt2RefSOPInstanceUID ;
end