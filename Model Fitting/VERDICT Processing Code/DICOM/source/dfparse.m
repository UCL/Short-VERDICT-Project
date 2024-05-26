function [ dinfo, UseDictionaryVR ] = dfparse(fns, verb, UseDictionaryVR)
%DFPARSE DICOM file parse creating info struct
% % Parses SingleFrame and MultiFrame (Enhanced) DICOM to create a
% structure dinfo that is the same whether input is Enhanced or Classic.
% The structure contians limited information (the original filename is
% stored so more data can be retrieved later if needed).
%
%   Calls cast_private to fix some PACs and transfer issues.
%   Calls data_process to perform some checks
%
% dinfo = dfparse(fns)
% dinfo = dfparse(fns, verbose)
% dinfo = dfparse(fns, verbose, UseDictionaryVR)
% [dinfo, UseDictionaryVR] = dfparse ...
%
% fns is cell array of file names or a single character array.
% verbose {true} | false 
%
% dinfo is a structure array with key fields, including an integer 
% slice index in field sl
%
% UseDictionaryVR is true/false after reading (to avoid 1000's of warnings)
%
% The code is subject to continued modification to cope with differences
% between vendors' DICOMs.
%
% Note TemporalPositionIdentifier is returned, whereas MultiFrame uses
% TemporalPositionIndex. These are assumed to be the same here.
%
% DiffusionDirectionality is converted to numeric:
%  0 NONE, 1 DIRECTIONAL, 2 ISOTROPIC
%
% See also DSELECTOR DPARSE D2MAT CAST_PRIVATE DATA_PROCESS
%
% Copyright, 2021-2023, David Atkinson.
% D.Atkinson@ucl.ac.uk
%

% This code is a combination of previously separate code dmfparse and
% dparse

if nargin < 2 || isempty(verb)
    verb = true ;
end

if nargin < 3 || isempty(UseDictionaryVR)
    UseDictionaryVR = false ; % the default for dicominfo
end

% These DICOM 3 tags will be placed in dinfo if present in single frame files
tags = {'Filename', 'PixelSpacing', 'ImageOrientationPatient', ...
    'ImagePositionPatient', 'Height', 'Width', 'SeriesNumber', ...
    'SliceThickness', 'AcquisitionTime', 'SeriesTime', ...
    'TemporalPositionIdentifier', ...       % Philips (helpful)
    'ImageType', ...  % Useful for Dixon
    'InversionTime',...
    'TriggerTime', ...
    'FlipAngle', 'Private_2001_1023', ... % 1st can lose precision
    'RepetitionTime', 'Private_2005_1030', ...  % 1st can be zero if small!
    'DiffusionBValue', 'DiffusionGradientOrientation', ...
    'Private_0019_100c', ...  % Siemens b-value
    'Private_0019_100d', ... % Siemens Diffusion Directionality
    'ProtocolName', ...
    'StudyInstanceUID', ...
    'FrameOfReferenceUID', ...
    'MRAcquisitionType',...
    'AcquisitionNumber',...
    'EchoNumber', ...
    'EchoNumbers', ...
    'EchoTime',...
    'RescaleSlope','RescaleIntercept', 'Private_2005_100e', ...
    'RealWorldValueMappingSequence', ...
    'Private_2005_1429',... % Philips ASL label type for single frame images
    'Private_2001_1008',... % Philips ASL phase number for multiphase images
    'Private_2001_1017',... % Philips ASL number of phases for multiphase images
    'Private_2005_1409','Private_2005_140a', ... % DA RescaleSlope and Intercept
    'Private_0043_1030', ... % GE VasCollapseFlag (Diffusion info)
} ;


dinfo = struct ;
id = 0 ;

if ischar(fns)
    cellfns{1} = fns ;
    fns = cellfns ;
end

ndf = length(fns) ;

if verb, hw = waitbar(0,'Processing files');  end 

for idf = 1:ndf

    if ~isfile(fns{idf})
        warning('File not available (Hard Drive issue?')
    end
    
    pssw = 0 ; % Private Scale Slope warning counter (multiframe)

    info = dicominfo(fns{idf},'UseDictionaryVR',UseDictionaryVR) ;

    if isfield(info,'ConfidentialityCode') && UseDictionaryVR==false 
        % Setting below to prevent 1000's of warnings from badly formed
        % DICOMS
        warning('dfparse: Setting UseDictionaryVR to true')
        UseDictionaryVR = true ;
    end

    if ~isfield(info,'SOPClassUID')
        % Maybe DICOMDIR file
        continue
    end
    
    switch info.SOPClassUID
        case '1.2.840.10008.5.1.4.1.1.7.4' % Multi-frame true colour sec capture
            dmfinfo = info ;
            nd = dmfinfo.NumberOfFrames ;
            fn = dmfinfo.Filename;
            Width = dmfinfo.Width ;
            Height = dmfinfo.Height ;
            SeriesNumber = dmfinfo.SeriesNumber ;
            if isfield(dmfinfo,'ProtocolName')
                ProtocolName = dmfinfo.ProtocolName ;
            else
                ProtocolName = '';
            end

            StudyDate = dmfinfo.StudyDate ;
            StudyTime = dmfinfo.StudyTime ;

            for iframe = 1:nd  % looping through frames
                if verb, waitbar(iframe/(nd*ndf),hw), end ;

                id = id + 1 ;
                dinfo(id).Frame = iframe ;
                dinfo(id).Filename = fn ;
                dinfo(id).Width = Width ;
                dinfo(id).Height = Height ;
                dinfo(id).SeriesNumber = SeriesNumber ;
                dinfo(id).ProtocolName = ProtocolName ;
                dinfo(id).StudyDate = StudyDate ;
                dinfo(id).StudyTime = StudyTime ;
            end


        case '1.2.840.10008.5.1.4.1.1.4.1' % Enhanced MR
            dmfinfo = info ;

            nd = dmfinfo.NumberOfFrames ;

            fn = dmfinfo.Filename;
            Width = dmfinfo.Width ;
            Height = dmfinfo.Height ;
            SeriesNumber = dmfinfo.SeriesNumber ;
            if isfield(dmfinfo,'ProtocolName')
                ProtocolName = dmfinfo.ProtocolName ;
            else
                ProtocolName = '';
            end
            FrameOfReferenceUID = dmfinfo.FrameOfReferenceUID ;
            StudyDate = dmfinfo.StudyDate ;
            StudyTime = dmfinfo.StudyTime ;
            StudyInstanceUID = dmfinfo.StudyInstanceUID ;

            if isfield(dmfinfo,'MRAcquisitionType')
                MRAcquisitionType = dmfinfo.MRAcquisitionType ;
            end

            % Number of ASL phases
            if isfield(dmfinfo,'Private_2001_1017')
                numASLPhases = dmfinfo.Private_2001_1017;
            end

            % analyse dimensions (needed for diffusion direction index)
            dimDGO = [] ; % dimension of Diffusion Gradient Orientation
            if isfield(dmfinfo,'DimensionIndexSequence')
                dis = dmfinfo.DimensionIndexSequence ;
                ndimind = length(fieldnames(dis)) ;
                for iind = 1:ndimind
                    itemstr = sprintf('Item_%d',iind) ;
                    disi = getfield(dis,itemstr) ;
                    if isfield(disi,'DimensionDescriptionLabel')
                        ddl = disi.DimensionDescriptionLabel ; % Not in Siemens?

                        switch ddl
                            case 'Diffusion Gradient Orientation'
                                dimDGO = iind ;
                        end
                    end
                end
            end

            fgs = dmfinfo.PerFrameFunctionalGroupsSequence ;
            sfgs = dmfinfo.SharedFunctionalGroupsSequence ;

            for iframe = 1:nd  % looping through frames
                if verb, waitbar(iframe/(nd*ndf),hw), end ;

                itemstr = sprintf('Item_%d',iframe) ;

                fg =  getfield(fgs,itemstr) ;
                sfg = getfield(sfgs, 'Item_1') ;

                id = id + 1 ;
                dinfo(id).Frame = iframe ;
                dinfo(id).Filename = fn ;
                dinfo(id).Width = Width ;
                dinfo(id).Height = Height ;
                dinfo(id).SeriesNumber = SeriesNumber ;
                dinfo(id).ProtocolName = ProtocolName ;
                dinfo(id).FrameOfReferenceUID = FrameOfReferenceUID ;
                dinfo(id).StudyDate = StudyDate ;
                dinfo(id).StudyInstanceUID = StudyInstanceUID ;
                dinfo(id).StudyTime = StudyTime ;
                if exist('MRAcquisitionType','var')
                    dinfo(id).MRAcquisitionType = MRAcquisitionType ;
                end

                %IOP  IPP
                dinfo(id).ImageOrientationPatient = fg.PlaneOrientationSequence.Item_1.ImageOrientationPatient;

                dinfo(id).ImagePositionPatient = fg.PlanePositionSequence.Item_1.ImagePositionPatient ;

                % Diffusion
                % ADC DICOM seem to have MRDiffusionSequence with the only sub-field
                % being DiffusionDirectionality 'NONE'
                if isfield(fg,'MRDiffusionSequence')
                    fgd = fg.MRDiffusionSequence.Item_1 ;
                    if isfield(fgd,'DiffusionBValue')
                        dinfo(id).DiffusionBValue = fgd.DiffusionBValue ;
                    end
                    if isfield(fgd,'DiffusionDirectionality')
                        ddty = fgd.DiffusionDirectionality ;
                        switch ddty
                            case 'ISOTROPIC'
                                dinfo(id).DiffusionDirectionality = 2 ;
                            case 'DIRECTIONAL'
                                dinfo(id).DiffusionDirectionality = 1 ;
                            case 'NONE'
                                dinfo(id).DiffusionDirectionality = 0 ;
                            otherwise
                                warning(['Diff Directionality problems'])
                                dinfo(id).DiffusionDirectionality = 0 ;
                        end
                    end

                    if isfield(fgd,'DiffusionGradientDirectionSequence')
                        dinfo(id).DiffusionGradientOrientation = fg.MRDiffusionSequence.Item_1.DiffusionGradientDirectionSequence.Item_1.DiffusionGradientOrientation ;
                        if ~isempty(dimDGO)
                            div = fg.FrameContentSequence.Item_1.DimensionIndexValues ;
                            dinfo(id).DiffGradOrientIdentifier = div(dimDGO) ;
                        end
                    end
                end

                dinfo(id).PixelSpacing = fg.PixelMeasuresSequence.Item_1.PixelSpacing ;
                if isfield(fg.PixelMeasuresSequence.Item_1,'SliceThickness')
                    dinfo(id).SliceThickness = fg.PixelMeasuresSequence.Item_1.SliceThickness ;
                else
                    dinfo(id).SliceThickness = dmfinfo.SpacingBetweenSlices ;
                end
                dinfo(id).RescaleSlope = fg.PixelValueTransformationSequence.Item_1.RescaleSlope ;
                dinfo(id).RescaleIntercept = fg.PixelValueTransformationSequence.Item_1.RescaleIntercept ;
                if isfield(fg,'Private_2005_140f') && isfield(fg.Private_2005_140f,'Item_1')
                    dinfo(id).Private_2005_100e = typecast(fg.Private_2005_140f.Item_1.Private_2005_100e,'single') ;
                else
                    % scale slope
                    pssw = pssw + 1;
                    dinfo(id).Private_2005_100e = 1 ;
                end

                dinfo(id).TemporalPositionIdentifier = fg.FrameContentSequence.Item_1.TemporalPositionIndex ;
                if isfield(fg,'CardiacTriggerSequence') & ...
                        isfield(fg.CardiacTriggerSequence.Item_1,'CardiacTriggerDelayTime')
                    dinfo(id).CardiacTriggerDelayTime = fg.CardiacTriggerSequence.Item_1.CardiacTriggerDelayTime ;
                end

                if isfield(sfg, 'MRTimingAndRelatedParametersSequence')
                    dinfo(id).FlipAngle = sfg.MRTimingAndRelatedParametersSequence.Item_1.FlipAngle ;
                    dinfo(id).RepetitionTime = sfg.MRTimingAndRelatedParametersSequence.Item_1.RepetitionTime ;
                    TRset = true ;
                else
                    TRset = false ;
                end

                if isfield(sfg, 'MRReceiveCoilSequence')
                    dinfo(id).ReceiveCoilName = sfg.MRReceiveCoilSequence.Item_1.ReceiveCoilName ;
                end

                if isfield(sfg, 'MRAveragesSequence')
                    dinfo(id).NumberOfAverages = sfg.MRAveragesSequence.Item_1.NumberOfAverages ;
                end

                if isfield(sfg, 'MRImagingModifierSequence')
                    dinfo(id).ImagingFrequency = sfg.MRImagingModifierSequence.Item_1.TransmitterFrequency ;
                end


                if isfield(fg,'Private_2005_140f') && isfield(fg.Private_2005_140f,'Item_1')
                    dinfo(id).ImageType = fg.Private_2005_140f.Item_1.ImageType ;
                    if isfield(fg.Private_2005_140f.Item_1, 'EchoNumber')
                        dinfo(id).EchoNumber = fg.Private_2005_140f.Item_1.EchoNumber ;
                    elseif isfield(fg.Private_2005_140f.Item_1, 'EchoNumbers')
                        dinfo(id).EchoNumbers = fg.Private_2005_140f.Item_1.EchoNumbers ;
                    end
                    dinfo(id).InversionTime = fg.Private_2005_140f.Item_1.InversionTime ;

                    % pCASL   'LABEL' or 'CONTROL'
                    if isfield(fg.Private_2005_140f.Item_1, 'Private_2005_1429')
                        dinfo(id).MRImageLabelType = fg.Private_2005_140f.Item_1.Private_2005_1429;
                        if isa(dinfo(id).MRImageLabelType,'numeric')
                            str_mrilt = char(dinfo(id).MRImageLabelType) ;
                            dinfo(id).MRImageLabelType = str_mrilt(1:end-1)' ;
                        end
                    end
                    % ASL multi phase   'ASL Phase'
                    if isfield(fg.Private_2005_140f.Item_1, 'Private_2001_1008')
                        dinfo(id).Private_2001_1008 = fg.Private_2005_140f.Item_1.Private_2001_1008;
                        dinfo(id).Private_2001_1017 = numASLPhases;
                    end

                    % Philips Private Fields to enable DICOMs to be written
                    % out suitable for reading back into the scanner
                    if isfield(fg.Private_2005_140f.Item_1, 'Private_2001_100a') 
                        dinfo(id).ImagePlaneNumber = fg.Private_2005_140f.Item_1.Private_2001_100a ;
                    end
                    if isfield(fg.Private_2005_140f.Item_1, 'Private_2005_1011')
                        dinfo(id).MRImageTypeMR = fg.Private_2005_140f.Item_1.Private_2005_1011 ;
                    end
                    if isfield(fg.Private_2005_140f.Item_1, 'Private_2005_106e')
                        dinfo(id).MRImageScanningSequencePrivate = fg.Private_2005_140f.Item_1.Private_2005_106e ;
                    end


                end

                % Observed in Philips multiframe FFE
                if isfield(fg,'MRImageFrameTypeSequence')
                    dinfo(id).ImageType = fg.MRImageFrameTypeSequence.Item_1.FrameType ;
                end

                % Found case where TR in Private field was a 2 element structure
                % with 0 and correct value. Favour TR from main fields.
                if TRset == false & isfield(dmfinfo, 'Private_2005_1030') % TR in ms
                    dinfo(id).RepetitionTime = typecast(dmfinfo.Private_2005_1030,'single') ;
                end

                if isfield(fg,'MREchoSequence')
                    dinfo(id).EffectiveEchoTime = fg.MREchoSequence.Item_1.EffectiveEchoTime ;
                end

                if pssw == 1 
                    warning('MATLAB:dfparse:PrivateScaleSlopeNotPresent', ...
                        ['Private Scale Slope factor not found, value set to 1 in series: ', ...
                        num2str(SeriesNumber), ' (',ProtocolName,')'])
                end

            end

        case {'1.2.840.10008.5.1.4.1.1.4'  , '1.2.840.10008.5.1.4.1.1.2'}  % Classic MR or Classic CT
            if verb, waitbar(idf/ndf,hw), end ;

            id = id + 1;
            for itag = 1:length(tags)
                if isfield(info,tags{itag})
                    if strcmp(tags{itag},'AcquisitionTime') || strcmp(tags{itag},'SeriesTime')
                        dinfo = setfield(dinfo,{id},tags{itag},str2num(getfield(info,tags{itag}))) ;
                    else
                        dinfo = setfield(dinfo,{id},tags{itag},getfield(info,tags{itag})) ;
                    end
                else
                    % warning([tags{itag},' is not in info'])
                end
            end

            % Handle Philips Classic
            if isfield(dinfo(id), 'DiffusionBValue') && isfield(dinfo(id),'DiffusionGradientOrientation')
                if dinfo(id).DiffusionBValue > 0
                    if norm(dinfo(id).DiffusionGradientOrientation) == 0 
                        % Must be trace weighted?
                        dinfo(id).DiffusionDirectionality = 2 ;
                    else
                        % Must be individual direction?
                        dinfo(id).DiffusionDirectionality = 1 ;
                    end
                else
                    % b = 0
                    dinfo(id).DiffusionDirectionality = 0 ;
                end
            end

            % Handle GE Signa MR diffusion data (See end of this file)
            if isfield(dinfo(id), 'Private_0043_1030')
                vasflag = typecast(dinfo(id).Private_0043_1030,'uint16') ;
                switch vasflag
                    case {3,4,5}
                        % individual diffusion image
                        dinfo(id).DiffusionDirectionality = 1 ;
                    case 14
                        % b=0 image
                        dinfo(id).DiffusionDirectionality = 0 ;
                        dinfo(id).DiffusionBValue = 0 ;
                    case 15
                        % CMB (combined, trace weighted image)
                        dinfo(id).DiffusionDirectionality = 2 ;
                    otherwise
                        % not recognised / implemented
                end
            end

            if isfield(dinfo(id), 'RealWorldValueMappingSequence')
                dinfo(id).RealWorldValueIntercept = dinfo(id).RealWorldValueMappingSequence.Item_1.RealWorldValueIntercept;
                dinfo(id).RealWorldValueSlope = dinfo(id).RealWorldValueMappingSequence.Item_1.RealWorldValueSlope;
            end

            if ~isfield(dinfo,'ProtocolName')
                dinfo.ProtocolName = '';
            end

        otherwise % Not MR
    end % SOPClasUID
end  % idf

if verb, close(hw), end % close waitbar

% RealWorldIntercept and Slope have been put in dinfo.
if isfield(dinfo,'RealWorldValueMappingSequence')
    dinfo = rmfield(dinfo,'RealWorldValueMappingSequence') ;
end

% Values have been put into DiffusionDirectinality
if isfield(dinfo, 'Private_0043_1030')
    dinfo = rmfield(dinfo,'Private_0043_1030') ;
end

% PACs can mess up some Private fields, making them uint8. 
% cast_private recasts them to what they are expected to be
dinfo = cast_private(dinfo) ; 

% Checks planes parallel etc. (SOPClassUID absent if DICOMDIR file)
if isfield(info,'SOPClassUID')
    switch info.SOPClassUID
        case '1.2.840.10008.5.1.4.1.1.7.4' % Sec. capture - do nothing here
        otherwise
            dinfo = data_process(dinfo,[],verb) ;
    end
end


% - - - 
% GE Signa
% 
% (0043,1030) VasCollapseFlag
% 
% For your protocol I’d expect it to take these values for the different directions/image types:
% 3              R/L         (individual image from ALL)
% 4             A/P        (individual image from ALL)
% 5              S/I          (individual image from ALL)
% 14           T2           (b=0)
% 15           CMB      (combined image, trace weighted)
% 
% It can also take other values e.g. for oblique or TETRA images, as follows:
% 9              OR/L
% 10           OA/P
% 11           OS/I
% 16           TENSOR
% 43           Dir1        (individual image from TETRA)
% 44           Dir2        (individual image from TETRA)
% 45           Dir3        (individual image from TETRA)
% 46           Dir4        (individual image from TETRA)
