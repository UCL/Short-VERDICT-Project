function [ dinfo_out ] = dmfparse(ffn_in)
%DMFPARSE Multi-frame DICOM file parse 
%   Parses DICOM file 
%
% dinfo = dmfparse(filename)
% dinfo = dmfparse
%
% If filename is a cell array of filenames, they will all be read in and a
% concatentated dinfo output.
%
% dinfo is returned with key fields, including an integer slice index in
% field sl
%
% Note TemporalPositionIdentifier is returned, whereas MultiFrame uses
% TemporalPositionIndex. These are assumed to be the same here.
%
% Reads in some Private Fields for Philips data. Note Private data may have
% been 
% re-written as uint8 in file transfer and here are typecast to single.
%
% DiffusionDirectionality is converted to numeric:
%  0 NONE, 1 DIRECTIONAL, 2 ISOTROPIC
%
% Copyright, 2019, David Atkinson.
% D.Atkinson@ucl.ac.uk
%
% See also DATPARSE D2MAT XMLPARSE DPARSE
%

if nargin==1 
    if ~iscell(ffn_in)
        ffn{1} = ffn_in ;
    else
        ffn = ffn_in ;
    end
end

if nargin==0 || ~exist(ffn{1},'file')
  global DMFFN   % ensures name persist across calls to this 
               % function (for ease of use and testing, not essential)
                        
  [fn,pn] = uigetfile({'*'}, 'Select MF DICOM file', DMFFN) ;
  if isequal(fn, 0) ; dinfo = [] ; return; end ;
  DMFFN = fullfile(pn,fn) ;
 
  ffn{1} = DMFFN ;
end

dinfo_out = [] ;


for ifile = 1:length(ffn)
    disp(['Loading info for file: ',ffn{ifile}, ' ...'])
    
    pssw = 0 ; % Private Scale Slope warning counter
    
    dmfinfo = dicominfo(ffn{ifile}) ;
    if ~isfield(dmfinfo,'NumberOfFrames')
        % Not multi-frame - call dparse
        % Probably should base the test on SOPClassUID: 
        %   1.2.840.10008.5.1.4.1.1.4.1	Enhanced MR Image Storage
        [pthn] = fileparts(ffn{ifile}) ;
        dinfo_out = dparse(pthn) ;
        return
    end
    if isfield(dmfinfo,'DeidentificationMethod')
        disp(['DeidentificationMethod: ',dmfinfo.DeidentificationMethod])
    end
    
    nd = dmfinfo.NumberOfFrames ;
    
    % Needs setting correctly (not enough fields here):
    dinfo = struct('RescaleSlope',num2cell(ones([1 nd])), ...
        'RescaleIntercept', num2cell(zeros([1 nd])), ...
        'Private_2005_100e',num2cell(ones([1 nd])), ...
        'sl',[],'TemporalPositionIdentifier',[], ...
        'PixelSpacing', [], 'Width',[],'Height',[], ...
        'SliceThickness', [], ...
        'FlipAngle', [], ...
        'DiffusionBValue',[], ...
        'DiffusionGradientOrientation', [], ...
        'DiffGradOrientIdentifier', [], ...
        'SeriesNumber',[],'ProtocolName',[],...
        'MRImageLabelType','', ...
        'Frame',[] ) ;
    
    % tags copied to all (bit wasteful but simplifies processing and potentially
    % allows multi-patient comparison
    
    fn = dmfinfo.Filename;
    Width = dmfinfo.Width ;
    Height = dmfinfo.Height ;
    SeriesNumber = dmfinfo.SeriesNumber ;
    ProtocolName = dmfinfo.ProtocolName ;
    FrameOfReferenceUID = dmfinfo.FrameOfReferenceUID ;
    if isfield(dmfinfo,'MRAcquisitionType')
      MRAcquisitionType = dmfinfo.MRAcquisitionType ;
    end
    
    % MJS adding a field for number of ASL phases
    if isfield(dmfinfo,'Private_2001_1017')
      numASLPhases = dmfinfo.Private_2001_1017;
    end

    % analyse dimensions (needed for diffusion direction index)
    dimDGO = [] ; % dimension of Diffusion Gradient Orientation
    dis = dmfinfo.DimensionIndexSequence ;
    ndimind = length(fieldnames(dis)) ;
    for iind = 1:ndimind
        itemstr = sprintf('Item_%d',iind) ;
        disi = getfield(dis,itemstr) ;
        ddl = disi.DimensionDescriptionLabel ;
        
        switch ddl
            case 'Diffusion Gradient Orientation'
                dimDGO = iind ;
        end
    end
    
    
    
    hw = waitbar(0,['Processing ',num2str(nd),' frames in file']) ;
    
    fgs = dmfinfo.PerFrameFunctionalGroupsSequence ;
    sfgs = dmfinfo.SharedFunctionalGroupsSequence ;
    
    for id = 1:nd
        waitbar(id/nd,hw) ;
        
        itemstr = sprintf('Item_%d',id) ;
        
        fg = getfield(fgs,itemstr) ;
        sfg = getfield(sfgs, 'Item_1') ;
        
        dinfo(id).Frame = id ;
        dinfo(id).Filename = fn ;
        dinfo(id).Width = Width ;
        dinfo(id).Height = Height ;
        dinfo(id).SeriesNumber = SeriesNumber ;
        dinfo(id).ProtocolName = ProtocolName ;
        dinfo(id).FrameOfReferenceUID = FrameOfReferenceUID ;
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
        if isfield(fg,'Private_2005_140f')
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
        
        if isfield(fg,'Private_2005_140f')
            dinfo(id).ImageType = fg.Private_2005_140f.Item_1.ImageType ;
            % Newer Philips now has EchoNumbers instead of EchoNumber d2mat
            % will now handle either
            if isfield(fg.Private_2005_140f.Item_1, 'EchoNumber')
                dinfo(id).EchoNumber = fg.Private_2005_140f.Item_1.EchoNumber ;
            else
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
            % MJS ASL multi phase   'ASL Phase' 
            if isfield(fg.Private_2005_140f.Item_1, 'Private_2001_1008')
                dinfo(id).Private_2001_1008 = fg.Private_2005_140f.Item_1.Private_2001_1008;
                dinfo(id).Private_2001_1017 = numASLPhases;
            end

        end
        
        % Found case where TR in private field was a 2 element structure
        % with 0 and correct value. Favour TR from main fields.
        if TRset == false & isfield(dmfinfo, 'Private_2005_1030') % TR in ms
            dinfo(id).RepetitionTime = typecast(dmfinfo.Private_2005_1030,'single') ;
        end
        
        if isfield(fg,'MREchoSequence')
            dinfo(id).EffectiveEchoTime = fg.MREchoSequence.Item_1.EffectiveEchoTime ;
        end
    end
    close(hw), drawnow  % close waitbar
    
    
    disp(['Processed ',num2str(nd),' DICOM frames'])
    
    if pssw > 0
        warning(['Private Scale Slope factor not found in data - value set to 1.'])
    end
    
    dinfo = data_process(dinfo) ; % check slices etc
    
    dinfo_out = [dinfo_out  dinfo] ;
end % ifile



end

