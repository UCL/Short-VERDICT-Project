%dicomRTStruct Handle ROIs information for DICOM-RT Structure Set
%
% Adaptation of MathWorks dicomContours to allow input of Ref meta data so
% that the created DICOM RTSTRUCT has FrameOfReferenceUID, PatientName etc
%
% Usage:
%  refInfo = dicominfo('enhanced.dcm') ;
%  % OR
%  refInfo = infoFromClassic(fns) ; % fns is cell array of Classic filenames
%    % e.g. from dselector
%
%  rtcont  = dicomRTStruct(refInfo) ;  % will have 0 ROIs
%     contours ...
%  rtcont  = addContour(rtcont,num,name,contours,'CLOSED_PLANAR',[255 0 0])
%  % repeat adding, deleting, plot etc
%  RTInfo = convertToInfo(rtcont) ;
%  status = dicomwrite([],'rtfile.dcm',RTInfo,'CreateMode','copy') 
%
% BELOW IS FROM ORGINAL MATHWORKS dicomContours
%
% CONTOURS = dicomContours(INFO) extracts ROIs information from DICOM-RT
% structure set and stores as dicomContours object. The extracted ROIs
% information is stored as a table in the ROIs property of the
% dicomContours object. The dicomContours object also stores the metadata
% of DICOM-RT structure set as a hidden property.
%
% dicomContours properties:
%    ROIs - Table containing the extracted ROIs information. (Read Only)
%
% dicomContours methods:
%    dicomContours - Construct dicomContours object
%    addContour    - Add ROI data to dicomContours object
%    deleteContour - Remove ROI data from dicomContours object
%    convertToInfo - Create dicom metadata with ROIs information in dicomContours object.
%    plotContour   - Plot ROIs contour data
%    createMask    - Create dense binary ROI representative of a set of contours in a user defined image space. 
%    
%    
%
%
% Example
% -------
% % Read metadata of DICOM-RT Structure Set
% info = dicominfo('rtstruct.dcm');
%
% % Construct dicomContours object
% rtContours = dicomContours(info);
%
% % Display all ROIs information as a table
% rtContours.ROIs
%
% % Plot contours of all ROIs
% plotContour(rtContours)
%
% % Add new ROI
% load contours
% rtContours = addContour(rtContours,3,'organ',contours,'closed_planar',[255,0,0]);
%
% newInfo = convertToInfo(rtContours);
% dicomwrite([],'newrtfile.dcm',newInfo,'createmode','copy')
%
% See also dicominfo, dicomwrite

% Copyright 2019-2020 The MathWorks, Inc.

classdef dicomRTStruct
    
    properties (Access = private)
        RefMetadata % meta data for the reference images (can be array for single frame)
        
        OrigMetadata % meta data for RTStruct file (Orig refers to before add/delete contour)
        Dictionary = dicomdict('get_current');      
    end
    
    properties (SetAccess = private, GetAccess = public)
        
        %ROIs - Table containing the extracted ROI information
        %
        % ROIs is a table of details with one row for each ROI. It contains 
        % the following columns:
        % Number        - ROI number, it is an integer.
        % Name          - Name of the ROI, it is a character vector.
        % ContourData   - ROI contours, stored as a cell array. Each
        %                 cell contains an N-by-3 array of the (X,Y,Z)
        %                 vertices of the contour specified in the patient
        %                 coordinate system.
        % GeometricType - Geometric type of the contour is a cell array of
        %                 character vector. Acceptable geometric type
        %                 character vectors are 'POINT', 'CLOSED_PLANAR',            
        %                 'OPEN_PLANAR' and 'OPEN_NONPLANAR'.
        % Color         - Color of the ROI, it is a 3-by-1 vector.
        %                 The values lie in the range 0 to 255.  
        ROIs = array2table(double.empty(0,5),...
            'VariableNames',{'Number','Name','ContourData','GeometricType','Color'});
    end
    
    methods        
        function obj = dicomRTStruct(RefMetadata, varargin)
            %dicomContours Construct dicomContours object
            %
            % rtContours = dicomContours(info) takes the metadata of
            % DICOM-RT Structure Set(info) from dicominfo and constructs a
            % dicomContours object
            
            % Validate RefMetadata and create metadata (for RTStruct)
            metadata = validateMetadata(RefMetadata, obj.Dictionary, varargin{:});
            
            obj.RefMetadata = RefMetadata ;


            % Store metadata
            obj.OrigMetadata = metadata;
            % Get Structure Set Sequence attribute name from dictionary
            structureSetSeq = getStructureSetSeqAttrName(obj.Dictionary);
            if(isfield(metadata,structureSetSeq))
                % Read ROIs information from metadata
                obj.ROIs = convertStructToTable(metadata, obj.Dictionary);
            else
                warning(message('images:dicomContours:emptyStructureSet'));
            end
        end
                
        function obj = addContour(obj, Number, Name, ContourData, GeometricType, varargin)
            %addContours Add ROI data to dicomContours object
            %
            % rtContours = addContours(rtContours,Number,Name,ContourData,GeometricType)
            % adds an ROI with Number, Name, ContourData, GeometricType to
            % the dicomContours object.
            %
            % rtContours = addContours(rtContours,Number,Name,ContourData,GeometricType,Color)
            % adds an ROI with Number, Name, ContourData, GeometricType,
            % and Color to the dicomContours object.
            %
            % Number        - ROI number, specified as a scalar integer.
            % Name          - Name of the  ROI, specified as a character vector
            %                 or string scalar.
            % ContourData   - ROI contours, specified as a cell array. Each
            %                 cell contains an N-by-3 array of the (X,Y,Z)
            %                 vertices of the contour specified in the patient
            %                 coordinate system.
            % GeometricType - Geometric type of the contour, specified as
            %                 one character vector for all contours or a 
            %                 cell array of character vector. The character
            %                 vector must be one of the following: 'POINT',
            %                 'CLOSED_PLANAR', 'OPEN_PLANAR' and 'OPEN_NONPLANAR'.
            % Color         - Color of the ROI, specified as a 3-by-1 vector.
            %                 The values must lie in the range of 0 to 255.
            
            narginchk(5,6);
            if(nargin == 5)
                Color = [];
            else
                Color = varargin{1};
            end
            % Validate input arguments
            GeometricType = validateInputs(Number, Name, ContourData, GeometricType, Color);
            
            % Make sure that given ROI number does not exist in object
            if(~isempty(obj.ROIs) && any(obj.ROIs.Number == Number))
                error(message('images:dicomContours:existingROI',num2str(Number)));
            end
            Name = char(Name);
            
            % If given geometric type is a character array, repeat it 
            % for all contours.
            if((numel(GeometricType) == 1) && ischar(GeometricType{1}))
                GeometricType = repmat(GeometricType,numel(ContourData),1);
            end
            % Convert as column arrays
            if(isrow(Color))
                Color = Color';
            end
            if(isrow(GeometricType))
                GeometricType = GeometricType';
            end
            if(isrow(ContourData))
                ContourData = ContourData';
            end
            
            % Add ROI data to the table  
            obj.ROIs = [obj.ROIs;{Number,Name,{ContourData},{GeometricType},Color}];
        end
               
        function obj = deleteContour(obj,roiNumbers)
            %deleteContour Remove ROI data from dicomContours object
            %
            % rtContours = deleteContour(rtContours,roiNumbers) removes
            % ROI data specified by roiNumbers from the dicomContours
            % object. The roiNumbers can be a scalar or vector denoting one
            % or multiple ROI data respectively.
            
            validateNumbers(roiNumbers);
            if(isempty(obj.ROIs))
                error(message('images:dicomContours:emptyTableToDelete'));
            end
            
            idx = ismember(obj.ROIs.Number,roiNumbers);
            % Make sure that given ROI numbers exist in object
            missingROI = setdiff(roiNumbers,obj.ROIs.Number(idx));
            if(~isempty(missingROI))
                error(message('images:dicomContours:deleteAbsentROIs',num2str(missingROI)));
            end
            
            obj.ROIs(idx,:) = [];
        end
        
        function newMetadata = convertToInfo(obj)
            %convertToInfo Create dicom metadata with ROIs information in
            % dicomContours object.
            %
            % newMetadata = convertToInfo(rtContours) Creates dicom metadata
            % with ROIs information in dicomContours object. The function
            % modifies ROIs information in the metadata with the ROI data in
            % dicomContours object. The metadata is stored as a hidden
            % property of the dicomContours object.

            % Get sequence attribute names from dicom dictionary 
            structureSetSeq = getStructureSetSeqAttrName(obj.Dictionary);
            roiContourSeq = getROIContourSeqAttrName(obj.Dictionary);
            observationSeq = getObservationSeqAttrName(obj.Dictionary);
            
            % Copy metadata without StructureSetROISequence, 
            % ROIContourSequence and RTROIObservationsSequence.
            newMetadata = obj.OrigMetadata;
            if(isfield(newMetadata,structureSetSeq))
                newMetadata = rmfield(newMetadata,structureSetSeq);
            end
            if(isfield(newMetadata,roiContourSeq))
                newMetadata = rmfield(newMetadata,roiContourSeq);
            end
            if(isfield(newMetadata,observationSeq))
                newMetadata = rmfield(newMetadata,observationSeq);
            end

            % Write ROI data of table into structure
            newMetadata = convertTableToStruct(obj,newMetadata);
        end
                
        function h = plotContour(obj,varargin)
            %plotContour Plot ROI contour data
            %
            % plotContour(rtContours) plots all ROIs contour data stored in 
            % the dicomContours object.
            %
            % plotContour(rtContour,roiNumbers) plots contour data of
            % specified ROI numbers, roiNumbers in dicomContours object. 
            % roiNumbers can be scalar or vector of integers.
            %
            % plotContour(____, HAX) plots ROIs in the given axes HAX.
            %
            % hg_Handle = plotContour(_____) returns hggroup handles to
            % plotted ROIs. hg_Handle can be one handle or handle array and 
            % each handle represent one ROI.
            
            narginchk(1, 3);
            
            if(isempty(obj.ROIs))
                error(message('images:dicomContours:emptyTableToPlot'));
            end
            [hax, roiNumbers] = parseInputs(varargin{:});

            if isempty(roiNumbers)
                roiNumbers = obj.ROIs.Number;
            end
            
            validateNumbers(roiNumbers);
            
            if ~isempty(hax)
                hax = newplot(hax);
            else
                hax = newplot;
            end
            
            % Verify whether any given ROI does not present in object
            idx = ismember(obj.ROIs.Number,roiNumbers);
            missingROI = setdiff(roiNumbers,obj.ROIs.Number(idx));
            if(~isempty(missingROI))
                error(message('images:dicomContours:plotAbsentROIs',num2str(missingROI)));
            end
            
            map = lines(numel(roiNumbers));
            hgHandle = gobjects(numel(roiNumbers),1);

            % Loop through ROIs and plot contour data
            for roiCount = 1:numel(roiNumbers)
                roiIndex = ismember(obj.ROIs.Number, roiNumbers(roiCount));
                % Get color
                color = map(roiCount,:);
                if(~isempty(obj.ROIs.Color{roiIndex}))
                    color = obj.ROIs.Color{roiIndex}/255;
                end
                
                hgHandle(roiCount) = hggroup(hax,'Tag',obj.ROIs.Name{roiIndex});
                contourData = obj.ROIs.ContourData{roiIndex};
                % Loop through contours
                for contourNum = 1:numel(contourData)
                    % Last point should be equal to first point for closed polygon geometry
                    if(isequal(obj.ROIs.GeometricType{roiIndex}{contourNum},'CLOSED_PLANAR'))
                        contourData{contourNum}(end+1,:) = contourData{contourNum}(1,:);
                    end
                    plot3(contourData{contourNum}(:,1),...
                        contourData{contourNum}(:,2),...
                        contourData{contourNum}(:,3),...
                        'color',color,'Parent',hgHandle(roiCount));
                end
            end
            if(nargout > 0)
                h = hgHandle;
            end
        end
        
        % This function takes in information about a 3D image, along with
        % the index of the ROI within this dicomContours object we wish to
        % use, and creates a dense mask representative of the ROI in the
        % intrinsic space of the 3D image.
        function BW = createMask(obj, ROIINDEX, spatial)
            %createMask Create volumetric mask from dicomContours object  
            % 
            % BW = createMask(rtContours,ROIINDEX,SPATIAL) creates BW, a
            % 3-D voxel representation of the ROI specified by ROIINDEX, in
            % the dicomContours object, rtContours. BW is 3-D logical mask.
            % ROIINDEX is a numerical value, character, or string which
            % specifies the contour in rtContours to be densified. SPATIAL
            % is a struct, returned by dicomReadVolume, containing the
            % fields: 'PatientPosition', 'PixelSpacing', and
            % 'PatientOrientation', or an IMREF3D object. The contour data
            % is in a world coordinate system; however, the mask is in the
            % intrinsic image space defined by SPATIAL.
            %
            % Class Support
            % -------------
            % rtContours is a dicomContours object. ROIINDEX is an integer,
            % character, or string. SPATIAL is a structure or an imref3d
            % object. BW is a 3-D logical array.
            %
            % Example
            % -------
            % % Use dicomInfo and imref3d to create a mask of a
            % % dicomContours object.
            %
            % % Read the metadata of a DICOM-RT Structure Set
            % info = dicominfo('rtstruct.dcm');
            %  
            % % Construct a dicomContours object
            % rtContours = dicomContours(info);
            %  
            % % Display all the ROI information as a table
            % rtContours.ROIs
            %  
            % % Plot the contours of all ROIs. This will plot the contours
            % % in world coordinates. The boundaries of this plot can be
            % % used to define world coordinate limit boundaries for an
            % % imref3d object to create a dense mask within.
            % plotContour(rtContours)
            % 
            % % Create an imref3d object with same world limits as plot
            % % from 'plotContours' so that the image is in the same space
            % % as the contours.
            % referenceInfo = imref3d([128, 128, 50], xlim, ylim, zlim);
            % 
            % % Specify the first contour ('Body_Contour') in the list of
            % % contours within rtContours.
            % contourIndex = 1;
            % 
            % % Create a 3-D logical mask using the imref3d objects and the
            % % contourIndex.
            % rtMask = createMask(rtContours, contourIndex, referenceInfo);
            % 
            % % View this mask using the Volume Viewer
            % volshow(rtMask);
            % 
            % See also dicominfo, dicomContours, dicomreadVolume, imref3d            
            
            if(~(isstruct(spatial) || isa(spatial, 'imref3d')))
               error(message('images:dicomContours:invalidImageInfo'));
            end
            
            validateattributes(ROIINDEX,{'numeric', 'char', 'string'},{},mfilename,'ROIINDEX',1);
            
            if (isa(ROIINDEX, 'numeric'))
                validateattributes(ROIINDEX,{'numeric'}, {'scalar', 'real', 'nonnan', 'finite'}, mfilename,'ROIINDEX',1);
                
            end
            
            BW = images.internal.dicom.createMaskHelp(obj, ROIINDEX, spatial);
  
        end
    end
end


function metadata = validateMetadata(RefMetadata,dictionary, varargin)
% Validate RefMetadata
validateattributes(RefMetadata, {'struct'},{'nonempty'}, mfilename, 'RefMetadata');
% Get attribute name from dicom dictionary. 
% (0008,0016) is known as 'SOPClassUID'.
uidAttrName = images.internal.dicom.lookupActions('0008','0016', dictionary);

if(~isfield(RefMetadata,uidAttrName))
    error(message('images:dicomRTStruct:musthaveSOPClassUID'));
end

if(isrtstruct(RefMetadata(1).(uidAttrName)))
    metadata = RefMetadata ; % RTStruct was passed in
else
    % Need to get RT attributes from RefMetadata
    metadata = ref2RT(RefMetadata, varargin{:}) ;
end
end


function tf = isrtstruct(actUID)

% Standard SOPClassUID for RT Structure Set
expUID = '1.2.840.10008.5.1.4.1.1.481.3';

tf = isequal(actUID, expUID);
end


function validateNumbers(roiNumbers)

validateattributes(roiNumbers, {'numeric'},...
    {'nonempty','finite','real','integer','nonsparse','vector'},...
    mfilename,'ROINumber');
end


function geometricType = validateInputs(number, name, points, geometricType, color)
% Validate ROI number
validateattributes(number, {'numeric'},...
    {'nonempty','finite','real','integer','nonsparse','scalar'},...
    mfilename,'ROINumber');
% Validate ROI name
if(~isempty(name))
    validateattributes(name,{'char','string'}, {'nonsparse'},mfilename,'ROIName');
end

% Validate contour data and it's geometric type
hasEmptyContour = isempty(points);
hasEmptyGeometric = isempty(geometricType);
% Make sure contour data and it's geometric type present 
if(xor(hasEmptyContour,hasEmptyGeometric))
    error(message('images:dicomContours:invalidContour'));
elseif(~(hasEmptyContour && hasEmptyGeometric))
    geometricType = validateContour(points,geometricType);
end

% Validate ROI color
if(~isempty(color))
    validateattributes(color,{'numeric'},...
        {'numel',3,'finite','real','integer','<=',255,'nonsparse','nonnegative'},...
        mfilename,'color');
end
end


function geometricType = validateContour(points,geometricType)

% GeometricType must be a char array or a cell array of size equal to ContourData.
validateattributes(points,{'cell'},{'vector'},mfilename,'ContourData');
validateattributes(geometricType,{'char','string','cell'},{'vector'},mfilename,'GeometricType');
if(ischar(geometricType))
    geometricType = {geometricType};
end
if(isstring(geometricType))
    geometricType = {char(geometricType)};
end
if(~any(numel(geometricType) == [1,numel(points)]))
    error(message('images:dicomContours:invalidSizeofGeometricType'));
end

% Acceptable geometric types 
allTypes = {'POINT','OPEN_PLANAR','OPEN_NONPLANAR','CLOSED_PLANAR'};
% Validate points of every contour 
for i = 1:numel(points)
    validateattributes(points{i},{'numeric'},...
        {'nonempty','ncols',3,'finite','nonsparse',},...
        mfilename,'Points of contour');
end
for i = 1:numel(geometricType)
   geometricType{i} = validatestring(geometricType{i},allTypes,mfilename,'GeometricType of Contour');
end
end


function rtTable = convertStructToTable(metadata, dictionary)

structureSetSeq = getStructureSetSeqAttrName(dictionary);
roiContourSeq = getROIContourSeqAttrName(dictionary);

% Get items in StructureSetROISequence 
items = fieldnames(metadata.(structureSetSeq));
numItems = numel(items);
% Intializing the variables for ROI table (Do not change the variable names).
Number = zeros(numItems,1);
Name = cell(numItems,1);
Color = cell(numItems,1);
ContourData = cell(numItems,1);
GeometricType = cell(numItems,1);

% Get items in ROIContourSequence 
[roiSeqItems,refROInum] = getROISeqItems(metadata, dictionary);
% Get attribute names from DICOM data dictionary
roiNumber = getroiNumberAttrName(dictionary);
roiName = getroiNameAttrName(dictionary);
roiColor = getroiColorAttrName(dictionary);
contourSequence = getContourSeqAttrName(dictionary);
contourDataField = getContourDataAttrName(dictionary);
geometricTypeField = getGeometricTypeAttrName(dictionary);

% Loop through each item in StructureSetROISequence
for itemNum = 1:numItems
    % Make sure 'ROINumber' field does exist
    if(~isfield(metadata.(structureSetSeq).(items{itemNum}),roiNumber))
        error(message('images:dicomContours:MissingROIField',...
            'ROINumber',items{itemNum},'StructureSetROISequence'));
    end
    % Make sure 'ROIName' field does exist
    if(~isfield(metadata.(structureSetSeq).(items{itemNum}),roiName))
        error(message('images:dicomContours:MissingROIField',...
            'ROIName',items{itemNum},'StructureSetROISequence'));
    end
    % Read ROI Number and Name from Structure Set ROI Sequence
    Number(itemNum) = metadata.(structureSetSeq).(items{itemNum}).(roiNumber);
    Name{itemNum} = metadata.(structureSetSeq).(items{itemNum}).(roiName);
    
    if(~isempty(refROInum))
        % Get corresponding item of ROI number in ROIContourSequence
        refItem = roiSeqItems{refROInum == Number(itemNum)};
        if(any(refROInum == Number(itemNum)))
            % Check ContourSequence does exist or not.
            if(isfield(metadata.(roiContourSeq).(refItem),(contourSequence)))
                % Get items in ContourSequence
                subitems = fieldnames(metadata.(roiContourSeq).(refItem).(contourSequence));
                % Loop through each item in ContourSequence
                for subitemNum = 1:numel(subitems)
                    % Make sure ContourData and ContourGeometricType fields
                    % exist in metadata
                    if(isfield(metadata.(roiContourSeq).(refItem).(contourSequence).(subitems{subitemNum}),...
                            {contourDataField,geometricTypeField}))
                        ContourData{itemNum}{subitemNum,1} = ...
                            reshape(metadata.(roiContourSeq).(refItem).(contourSequence).(subitems{subitemNum}).(contourDataField),3,[])';
                        GeometricType{itemNum}{subitemNum,1} = ...
                            metadata.(roiContourSeq).(refItem).(contourSequence).(subitems{subitemNum}).(geometricTypeField);                       
                    end
                end
            end
            % Read color
            if(isfield(metadata.(roiContourSeq).(refItem),(roiColor)))
                Color{itemNum} = metadata.(roiContourSeq).(refItem).(roiColor);
            end
        end
    end
end
% Check any ROI is repeated
if(~isequal(numel(unique(Number)),numel(Number)))
    error(message('images:dicomContours:repeatedROIs'));
end
rtTable = table(Number,Name,ContourData,GeometricType,Color);
end


function newMetadata = convertTableToStruct(obj, newMetadata)

% Get attribute names from DICOM data dictionary
structureSetSeq = getStructureSetSeqAttrName(obj.Dictionary);
roiContourSeq = getROIContourSeqAttrName(obj.Dictionary);
observationSeq = getObservationSeqAttrName(obj.Dictionary);
roiNumber = getroiNumberAttrName(obj.Dictionary);
referencedROINum = getRefROInumberAttrName(obj.Dictionary);
roiName = getroiNameAttrName(obj.Dictionary);
roiColor = getroiColorAttrName(obj.Dictionary);
contourSequence = getContourSeqAttrName(obj.Dictionary);
contourDataField = getContourDataAttrName(obj.Dictionary);
geometricTypeField = getGeometricTypeAttrName(obj.Dictionary);
numberOfContourPoints = getNumberOfContourPointsAttrName(obj.Dictionary);

% Get items and ROI numbers from ROI Sequences
[structSeqItems,structSeqROInum] = getStructSeqItems(obj.OrigMetadata, obj.Dictionary);
[roiSeqItems,roiSeqnum] = getROISeqItems(obj.OrigMetadata, obj.Dictionary);
[obsSeqItems,obsSeqROInum] = getObsSeqItems(obj.OrigMetadata, obj.Dictionary);

items = strcat('Item_',string(1:size(obj.ROIs,1)));
% Loop through rows of table 
for rowNum = 1: size(obj.ROIs,1)    
    newItem = items(rowNum);    
    % Get corresponding item of ROI number
    structSeqItem = structSeqItems(structSeqROInum == obj.ROIs.Number(rowNum));
    roiSeqItem = roiSeqItems(roiSeqnum == obj.ROIs.Number(rowNum));
    rtroiSeqItem = obsSeqItems(obsSeqROInum == obj.ROIs.Number(rowNum));
    
    % Copy ROISequence fields, if ROI Number does exist in OrigMetadata
    if(~isempty(structSeqItem))
        newMetadata.(structureSetSeq).(newItem) =...
            obj.OrigMetadata.(structureSetSeq).(structSeqItem{1});
        if(~isempty(roiSeqItem))
            newMetadata.(roiContourSeq).(newItem) =...
                obj.OrigMetadata.(roiContourSeq).(roiSeqItem{1});
        end
        if(~isempty(rtroiSeqItem))
            newMetadata.(observationSeq).(newItem) =...
                obj.OrigMetadata.(observationSeq).(rtroiSeqItem{1});
        end
    end
    
    % Write ROI data into the structure
    newMetadata.(structureSetSeq).(newItem).(roiNumber) = obj.ROIs.Number(rowNum);
    newMetadata.(structureSetSeq).(newItem).(roiName) = obj.ROIs.Name{rowNum};
    newMetadata.(structureSetSeq).(newItem).FrameOfReferenceUID = obj.RefMetadata(1).FrameOfReferenceUID ;

    newMetadata.(roiContourSeq).(newItem).(referencedROINum) = obj.ROIs.Number(rowNum);
    newMetadata.(observationSeq).(newItem).(referencedROINum) = obj.ROIs.Number(rowNum);
    newMetadata.(roiContourSeq).(newItem).(roiColor) = obj.ROIs.Color{rowNum};
    
    % Make new contour sequence
    if(isfield(newMetadata.(roiContourSeq).(newItem),(contourSequence)))
        newMetadata.(roiContourSeq).(newItem) =...
            rmfield(newMetadata.(roiContourSeq).(newItem),(contourSequence));
    end
    contours = obj.ROIs.ContourData{rowNum};
    geometricType = obj.ROIs.GeometricType{rowNum};
    contourSeq = makeContourSeq(contours,geometricType,contourDataField,geometricTypeField,numberOfContourPoints);
    if(~isempty(contourSeq))
        newMetadata.(roiContourSeq).(newItem).(contourSequence) = contourSeq;
    end
end
end


function contourSeq = makeContourSeq(contours,geometricType,contourDataField,geometricTypeField,numberOfContourPoints)

contourSeq = [];
numberOfContours = numel(contours);
subitems = strcat('Item_',string(1:numberOfContours));
for contourNum = 1:numberOfContours
    subItem = subitems(contourNum); 
    contourSeq.(subItem).(numberOfContourPoints) = size(contours{contourNum}, 1);
    contourSeq.(subItem).(contourDataField) = reshape((contours{contourNum})',[],1);  
    contourSeq.(subItem).(geometricTypeField) = geometricType{contourNum};
end
end


function [structSeqItems,structSeqROInum] = getStructSeqItems(metadata, dictionary)

structSeqItems = [];
structSeqROInum = [];

structureSetSeq = getStructureSetSeqAttrName(dictionary);
roiNumber = getroiNumberAttrName(dictionary);

if(isfield(metadata,structureSetSeq))
    % Get all items from Structure Set Sequence 
    structSeqItems = fieldnames(metadata.(structureSetSeq));
    structSeqROInum = zeros(numel(structSeqItems),1);
    for itemNum = 1:numel(structSeqItems)
        % Make sure 'ROINumber' field does exist in each item of Structure
        % Set Sequence
        if(~isfield(metadata.(structureSetSeq).(structSeqItems{itemNum}),(roiNumber)))
            error(message('images:dicomContours:MissingROIField',...
                'ROINumber',structSeqItems{itemNum},'StructureSetROISequence'));
        end
        % Read ROI number 
        structSeqROInum(itemNum) = metadata.(structureSetSeq).(structSeqItems{itemNum}).(roiNumber);
    end
else
    warning(message('images:dicomContours:emptyStructureSet'));
end
end


function [roiSeqItems,refROInum] = getROISeqItems(metadata, dictionary)

refROInum = [];
roiSeqItems = [];

roiContourSeq = getROIContourSeqAttrName(dictionary);
referencedROINum = getRefROInumberAttrName(dictionary);

if(isfield(metadata,roiContourSeq))
    % Get all items from ROI Contour Sequence
    roiSeqItems = fieldnames(metadata.(roiContourSeq));
    refROInum = zeros(numel(roiSeqItems),1);
    for itemNum = 1:numel(roiSeqItems)
        % Make sure 'ReferencedROINumber' field does exist in each item of
        % ROI Contour Sequence
        if(~isfield(metadata.(roiContourSeq).(roiSeqItems{itemNum}),(referencedROINum)))
            error(message('images:dicomContours:MissingROIField',...
                'ROINumber',roiSeqItems{itemNum},(roiContourSeq)));
        end
        % Read ROI number
        refROInum(itemNum) = metadata.(roiContourSeq).(roiSeqItems{itemNum}).(referencedROINum);
    end
end
end


function [obsSeqItems,obsSeqROInum] = getObsSeqItems(metadata, dictionary)

obsSeqItems = [];
obsSeqROInum = [];

observationSeq = getObservationSeqAttrName(dictionary);
referencedROINum = getRefROInumberAttrName(dictionary);

if(isfield(metadata,observationSeq))
    % Get all items from ROI Observations Sequence
    obsSeqItems = fieldnames(metadata.(observationSeq));
    obsSeqROInum = zeros(numel(obsSeqItems),1);
    for itemNum = 1:numel(obsSeqItems)
        % Make sure 'ReferencedROINumber' field does exist in each item of
        % ROI Observations Sequence
        if(~isfield(metadata.(observationSeq).(obsSeqItems{itemNum}),(referencedROINum)))
            error(message('images:dicomContours:MissingROIField',...
                'ROINumber',obsSeqItems{itemNum},(observationSeq)));
        end
        % Read ROI number
        obsSeqROInum(itemNum) = metadata.(observationSeq).(obsSeqItems{itemNum}).(referencedROINum);
    end
end
end


function structureSetSeq = getStructureSetSeqAttrName(dictionary)
% Look through the dictionary and get (3006,0020) attribute name.
% (3006,0020) known as 'StructureSetROISequence'.
structureSetSeq = images.internal.dicom.lookupActions('3006', '0020', dictionary);
end


function roiNumber = getroiNumberAttrName(dictionary)
% Look through the dictionary and get (3006,0022) attribute name.
% (3006,0022) known as 'ROINumber'.
roiNumber = images.internal.dicom.lookupActions('3006', '0022', dictionary);
end


function roiName = getroiNameAttrName(dictionary)
% Look through the dictionary and get (3006,0026) attribute name.
% (3006,0026) known as 'ROIName'.
roiName = images.internal.dicom.lookupActions('3006', '0026', dictionary);
end


function roiContourSeq = getROIContourSeqAttrName(dictionary)
% Look through the dictionary and get (3006,0039) attribute name.
% (3006,0039) known as 'ROIContourSequence'.
roiContourSeq = images.internal.dicom.lookupActions('3006', '0039', dictionary);
end


function observationSeq = getObservationSeqAttrName(dictionary)
% Look through the dictionary and get (3006,0080) attribute name.
% (3006,0080) known as 'RTROIObservationsSequence'.
observationSeq = images.internal.dicom.lookupActions('3006', '0080', dictionary);
end


function referencedROINum = getRefROInumberAttrName(dictionary)
% Look through the dictionary and get (3006,0084) attribute name.
% (3006,0084) known as 'ReferencedROINumber'.
referencedROINum = images.internal.dicom.lookupActions('3006', '0084', dictionary);
end


function roiColor = getroiColorAttrName(dictionary)
% Look through the dictionary and get (3006,002A) attribute name.
% (3006,002A) known as 'ROIDisplayColor'.
roiColor = images.internal.dicom.lookupActions('3006', '002A', dictionary);
end


function contourSequence = getContourSeqAttrName(dictionary)
% Look through the dictionary and get (3006,0040) attribute name.
% (3006,0040) known as 'ContourSequence'.
contourSequence = images.internal.dicom.lookupActions('3006', '0040', dictionary);
end


function geometricTypeField = getGeometricTypeAttrName(dictionary)
% Look through the dictionary and get (3006,0042) attribute name.
% (3006,0042) known as 'ContourGeometricType'.
geometricTypeField = images.internal.dicom.lookupActions('3006', '0042', dictionary);
end


function numberOfContourPoints = getNumberOfContourPointsAttrName(dictionary)
% Look through the dictionary and get (3006,0046) attribute name.
% (3006,0046) known as 'NumberOfContourPoints'.
numberOfContourPoints = images.internal.dicom.lookupActions('3006', '0046', dictionary);
end


function contourDataField = getContourDataAttrName(dictionary)
% Look through the dictionary and get (3006,0050) attribute name.
% (3006,0050) known as 'ContourData'.
contourDataField = images.internal.dicom.lookupActions('3006', '0050', dictionary);
end


function [hax, roiNumbers] = parseInputs(varargin)

hax = [];
roiNumbers = [];
if (nargin >= 1) && (~isempty(varargin{1}))
    % Can be ROI number or axes handle
    try
        if isnumeric(varargin{1})
            roiNumbers = varargin{1};
        elseif ishghandle(varargin{1}) && strcmp(get(varargin{1},'type'),'axes')
            hax = varargin{1};
        else
            error(message('images:dicomContours:invalidInput',2));
        end
    catch
       error(message('images:dicomContours:invalidInput',2));
    end
end

if (nargin == 2) && (~isempty(varargin{2}))
    % Can be ROI number or axes handle
    try
        if isnumeric(varargin{2}) && isempty(roiNumbers)
            roiNumbers = varargin{2};
        elseif ishghandle(varargin{2}) && strcmp(get(varargin{2},'type'),'axes') && isempty(hax)
            hax = varargin{2};
        else
            error(message('images:dicomContours:invalidInput',3));
        end
    catch
        error(message('images:dicomContours:invalidInput',3));
    end
end
end