function [ dinfo ] = dsfread(fns)
%DSFREAD Single frame DICOM file reader
%   Calls cast_private to fix some PACs and transfer issues
%
% dinfo = dsfread(fns)
%  fns is full path to a single file, or, cell array of full file names
%
% dinfo is returned with key fields, including an integer slice index in
% field sl
%
% See also D2MAT XMLPARSE CAST_PRIVATE DATA_PROCESS DPARSE
%
% David Atkinson

if ~iscell(fns)
    temp = fns ;
    clear fns
    fns{1} = temp ;
end


% These DICOM 3 tags will be placed in dinfo if present in files
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
    'ProtocolName', ...
    'FrameOfReferenceUID', ...
    'MRAcquisitionType',...
    'AcquisitionNumber',...
    'EchoNumber', ...
    'EchoTime',...
    'RescaleSlope','RescaleIntercept', 'Private_2005_100e', ...
    'RealWorldValueMappingSequence', ...
    'Private_2005_1429',... % MJS adding the label type for single frame images
    'Private_2001_1008',... % MJS adding the phase number for multiphase images
    'Private_2001_1017',... % MJS adding the number of phases for multiphase images
    'Private_2005_1409','Private_2005_140a', ... % DA RescaleSlope and Intercept
} ;

df = 1 ; % field counter

% dinfo = struct('RescaleSlope',num2cell(ones([1 nd])), 'RescaleIntercept', ...
%     num2cell(zeros([1 nd])), 'Private_2005_100e',num2cell(ones([1 nd]))) ;
dinfo = struct ;

npsxx = 0 ;

nfn = length(fns) ;

hw = waitbar(0,['Processing ',num2str(nfn),' files.']) ;

for ifn = 1:nfn
    waitbar(ifn/nfn,hw) ;
    
    fn = fns{ifn} ;
    
    
    info = dicominfo(fn) ;
    
    % insert skip if View Forum XX or PS files, DICOMDIR, non-geom
    if ~isfield(info,'DirectoryRecordSequence') && ...  % DICOMDIR
            isfield(info,'ImagePositionPatient') && ...  % 
            strcmp(info.SOPClassUID, '1.2.840.10008.5.1.4.1.1.11.1') == 0 && ...
            strcmp(info.SOPClassUID, '1.2.840.10008.5.1.4.1.1.66') == 0
         
             % 1.2.840.10008.5.1.4.1.1.11.1 Grayscale Softcopy Presentation State
             % 1.2.840.10008.5.1.4.1.1.66   Raw data
        for itag = 1:length(tags)
            if isfield(info,tags{itag})
                if strcmp(tags{itag},'AcquisitionTime') || strcmp(tags{itag},'SeriesTime')
                    dinfo = setfield(dinfo,{df},tags{itag},str2num(getfield(info,tags{itag}))) ;
                else
                    dinfo = setfield(dinfo,{df},tags{itag},getfield(info,tags{itag})) ;
                end
            else
                % warning([tags{itag},' is not in info'])
            end
        end
%       if ~isfield(info,'Private_2005_100e')
%           % Not in input, default value of 1 is best replaced with 1/RescaleSlope
%           dinfo(df).Private_2005_100e = 1/dinfo(df).RescaleSlope ;
%       end
        if isfield(dinfo(df), 'RealWorldValueMappingSequence')
            dinfo(df).RealWorldValueIntercept = dinfo(df).RealWorldValueMappingSequence.Item_1.RealWorldValueIntercept;
            dinfo(df).RealWorldValueSlope = dinfo(df).RealWorldValueMappingSequence.Item_1.RealWorldValueSlope;
        end

        df = df + 1 ;
          
        
    else % XX or PS file test OR DICOMDIR
        npsxx = npsxx + 1 ;
    end
end % id loop

close(hw)  % close waitbar

ndf = df - 1 ; 
dinfo(ndf+1:end) = [] ;
if npsxx > 0 
    str = [' (ignored ',num2str(npsxx),' files).'];
else
    str = ['.'];
end
disp(['Processed ',num2str(ndf),' DICOM files', str])

if isfield(dinfo,'RealWorldValueMappingSequence')
    dinfo = rmfield(dinfo,'RealWorldValueMappingSequence') ;
end

dinfo = cast_private(dinfo) ; % Fixes unknown value representaton 
                              % for Private fields in some circumstances
dinfo = data_process(dinfo) ;


end


