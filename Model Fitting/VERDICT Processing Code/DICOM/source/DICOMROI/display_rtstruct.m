function varargout = display_rtstruct
% DISPLAY_RTSTRUCT Display RTStruct on associated images and output mask
% 
%          display_rtstruct
% bwmask = display_rtstruct
%
% Calls dselector first for an RTStruct filename, then to select the image
% DICOM(s).
%
% bwmask has size [ny nx nz nROI] where ny,nx,nz correspond to the 
% spatial sizes of the DICOM images and nROI is number of ROIs in 
% RTstruct. Note one ROI can contain multiple contours, for example a 
% whole prostate contoured over multiple slices.
%
% Only works correctly for the DICOMs:
%   One frame per slice e.g. T2W, T1W, b2000 (with no b=0)
%   FFE with multi-echo and modulus and phase
%   When 3D, will use slice centre spacing as slice thickness
%
% Placing contours on a different Series (with the same Frame Of Reference
% UID) is possible, but may run into trouble if the slice separation does not
% match the RTstruct, e.g. some slices getting overlapping contours. In
% this case, bwmask will take the area of all.
%
% See also dselector indexRT


disp('Select RTStruct file')
rtfile = getdfiles(dselector,'Select RTStruct file') ;
rtinfo = dicominfo(rtfile{1}) ;
rtContours = dicomContours(rtinfo) ;

% Show 3D plot with all contours
if isfield(rtinfo,'StructureSetName')
    tstr = rtinfo.StructureSetName ;
else
    tstr = 'rtstruct' ;
end
figure(Name=tstr)
plotContour(rtContours), axis equal, title(tstr,'Interpreter','none')
xlabel('Left'), ylabel('Posterior'), zlabel('Head')
ax = gca ;
ax.Color = [0.2 0.2 0.2];

% FrameOfRefUID
% See also checks within indexRT on Frame Of Reference UIDs
if ~isfield(rtinfo,'ReferencedFrameOfReferenceSequence') || ...
        isempty(rtinfo.ReferencedFrameOfReferenceSequence)
    warning('MATLAB:display_rtstruct:ReferencedFrameOfRefSeqMissing','Missing ReferencedFrameOfRefUID')
    ReferencedFOfRefUID = '' ;
else 
    ReferencedFOfRefUID = rtinfo.ReferencedFrameOfReferenceSequence.Item_1.FrameOfReferenceUID ;
end

% Read in DICOM volume and check FrameOfRefUID is same as rtcontours
rtContours.ROIs  % Displays table of ROIs on screen
disp('Select DICOM image file(s)')
dinfo = getdinfo(dselector,'Select image file(s)') ;

if ~isfield(dinfo,'FrameOfReferenceUID') 
    warning('MATLAB:display_rtstruct:NoFOfRefUID','No FrameOfRefUID in image data')
    FrameOfRefUID = '' ;
else
    FrameOfRefUID = dinfo.FrameOfReferenceUID ;
end

if ~strcmp(ReferencedFOfRefUID,FrameOfRefUID)
    warning('MATLAB:display_rtstruct:InconsistentFrameOfRefUID', ...
        'FrameOfReferenceUIDs for contours and image do not match. Possible geometrical error in display')
end

% Not clear that MIM RTstruct contain a record of which frame the ROI was
% drawn on in the case of multi-frame data. Potentially problematic for
% data with more than one frame per slice position (e.g. DCE, multi-echo).
%
% For now, handle on case-by-case basis.

nFrame = length(dinfo) ;
[vf,mf,locf] = d2mat(dinfo,{'frame'}) ;
nSlice = length(mf.geom) ;

if nSlice == nFrame  % one frame per slice
    [V,m, loc] = d2mat(dinfo,{'slice'},'op','fp') ;
elseif length(mf.effTEVec_indata) > 1 && length(mf.itypeVec_indata) > 1 && ...
        find(mf.itypeVec_indata==29)
    % FFE multi-echo and mag/phase
    % Use first echo and magnitude data for plot and basis for mask.
    echoNum = 1 ;
    [V,m] = d2mat(dinfo,{'slice','effTE','itype'},'itype',29,'op','fp') ;
    disp(['Using echo at time: ',num2str(m.effTEVec(echoNum))])
    V = V(:,:,:,echoNum) ; 
    tstr = [tstr,' DISPLAYED on MAG echo: ',num2str(echoNum),'/',num2str(length(mf.effTEVec_indata))];
elseif length(mf.effTEVec_indata) > 1 && length(mf.itypeVec_indata) == 1
    % multi-echo
    echoNum = 1 ;
    [V,m] = d2mat(dinfo,{'slice','effTE'},'op','fp') ;
    disp(['Using echo at time: ',num2str(m.effTEVec(echoNum))])
    V = V(:,:,:,echoNum) ; 
    tstr = [tstr,' DISPLAYED on echo: ',num2str(echoNum),'/',num2str(length(mf.effTEVec_indata))];
else
    error('dinfo data pattern not implemented')
end


[rt2Slice, slice2RT, rt2RefSOPInstanceUID] = indexRT(rtContours, m.geom, rtinfo) ;

bwmask = zeros([size(V,[1 2 3]) size(rtContours.ROIs,1)]) ;

figure(Name=tstr)
tiledlayout('flow','TileSpacing','none');
nSlices = size(V,3) ;
ax = zeros([nSlices 1]) ;

for iSlice = 1:nSlices
    
    rtinslice = slice2RT{iSlice} ; % Get ROI and Contour numbers in slice

    if ~isempty(rtinslice)
        % Only output slices that have a contour (otherwise too cluttered)
        ax(iSlice) = nexttile ;
        hImage = imshow(V(:,:,iSlice),[]) ;
        title(['Slice: ',num2str(iSlice)])
    end

    for iRT = 1:size(rtinslice,1)
        iROIRT     = rtinslice(iRT,1) ;
        iContour = rtinslice(iRT,2) ;

        % currently not doing anything with this (might be used to check
        % ROI is on correct DICOM)
        refuiddat = rt2RefSOPInstanceUID{iROIRT};
        refUID = refuiddat{iContour};

        contdat = rtContours.ROIs.ContourData{iROIRT} ;
        inputCoord3D = contdat{iContour} ;

        [roiCoord2D, distances] = coord2D3D(inputCoord3D,m.geom(iSlice)) ;

        % Create MATLAB 2D ROI
        ROI2D = images.roi.Polygon(ax(iSlice),'Position',roiCoord2D) ;
        ROI2D.Color = rtContours.ROIs.Color{iROIRT}/255 ;
        ROI2D.Label = rtContours.ROIs.Name{iROIRT} ;
        ROI2D.MarkerSize = 4 ;

        bw2D = createMask(ROI2D,hImage) ; % Create a 2D mask from this ROI
        bwmask(:,:,iSlice,iROIRT) = bwmask(:,:,iSlice,iROIRT) + bw2D ;
    end
end
linkaxes(ax(ax~=0))


if nargout > 0
    maxbw = max(bwmask(:)) ;
    lochigh = find(bwmask(:) > 1) ;
    if ~isempty(lochigh)
        warning('MATLAB:display_rtstruct:overlappingContours', ...
            ['Overlapping contours: mask max set to 1, (was: ',num2str(maxbw),')'])

        bwmask(lochigh) = 1 ;
    end

    disp(['Output bwmask has size: ',num2str(size(bwmask))])
    disp('Corresponding ROIs are: ')
    rtContours.ROIs

    varargout{1} = bwmask ;
end

return


% data_folder = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/RTstruct_Examples/ReImagineScreening_Luminal/2010034_post/2000-01__Studies' ;
% volume_dicom = 'Reim.Ws2.2010034_2010034_MR_2000-01-10_104250_._T2W.TSE.ax_n31__00000/1.3.6.1.4.1.5962.99.1.967497304.1162466301.1594400396888.382.0.dcm' ;
% rtstruct_dicom = 'Reim.Ws2.2010034_2010034_RTst_2000-01-10_104250_._T2.Contours_n1__00000/2.16.840.1.114362.1.12046989.24664026888.592346988.756.29.dcm' ;
% 
% inplaneDistThresh = 0.1 ;  % threshold (mm) for differences in contour to image plane.
% roiNotInSliceEdgeAlpha = 0.4 ;  % 2D EdgeAlpha if ROI not in slice
% roiNotInSliceFaceAlpha = 0.1 ;
% roiNotInSliceMarkerSize = 4 ;
% roiNotInSliceLabelAlpha = 0.5 ;
% 
% vslice = 12 ;  % slice of interest in the DICOM volume
% indexContourData = 3 ; % In this data, number 1 are contours for Whole Prostate
% roiInSlice = true ; % reset later if ROI not in slice or angulated

% % Get Whole Prostate Contours (there are 15 in this dataset)
% WPC = rtContours.ROIs.ContourData{indexContourData} ; 
% nWPC = length(WPC) ; 
% 
% % Find contour that is closest to slice vslice.
% % Frist compute distances
% c2slicedist = zeros([1 nWPC]) ;
% for icont = 1:nWPC
%     inputCoord3D = WPC{icont}; % contour coordinates in 3D
% 
%     [outputCoord2D, distances] = coord2D3D(inputCoord3D,m.geom(vslice)) ;
%     % if contour is in the plane of the slice, the distances should all be
%     % the same
%     if abs(max(distances)-min(distances)) > inplaneDistThresh
%         warning('Contour may not be in plane of image')
%         roiInSlice = false ;
%     end
%     c2slicedist(icont) = mean(distances) ;
% end
% 
% % Find contour closest to slice (minimum distance)
% [minDist, ind] = min((abs(c2slicedist))) ;
% 
% if minDist  > m.geom(vslice).SliceThickness
%     roiInSlice = false ;
% end
% disp(['Closest contour (',num2str(ind),') is distance ',num2str(minDist)])
% 
% % Get the 2D coordinates of this contour to be used to make a 2D ROI
% roicoord2D = coord2D3D(WPC{ind}, m.geom(vslice)) ;
% 
% % Show the slice and create a 2D MATLAB ROI from the contour's 2D coordinates
% figure(Name='slice with ROI')
% hImage = imshow(V(:,:,vslice),[]) ;  hold on
% ROI = images.roi.Polygon(gca,'Position',roicoord2D) ;
% ROI.Color = rtContours.ROIs.Color{indexContourData}/255 ;
% ROI.Label = rtContours.ROIs.Name{indexContourData} ;
% if ~roiInSlce
%     ROI.EdgeAlpha  = roiNotInSliceEdgeAlpha ;
%     ROI.FaceAlpha  = roiNotInSliceFaceAlpha ;
%     ROI.MarkerSize = roiNotInSliceMarkerSize ;
%     ROI.LabelAlpha = roiNotInSliceLabelAlpha ;
% end
% 
% bw = createMask(ROI,hImage) ; % Create a mask from this ROI
% 
% figure(Name='Mask from ROI') % Display the mask
% imshow(bw)



