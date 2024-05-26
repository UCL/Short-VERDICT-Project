function dicoman(varargin)
% DICOMAN DICOM anonymisation keeping only specified information.
% NOTE: 1) May remove useful information.
%       2) May not comply with some guidelines because it keeps the
%          study and series descriptions.
%
% Anonymises all DICOM files in one folder and sub-folders one level down.
% Note no DICOMDIR file is written out.
% Maintains filenames and folder structure (names can be adjusted to
% contain slice information etc).
% Assumes that a dir lists the files by series so that one dicomuid can
% be used for all files in one series.
% Will insert the date that this programme is run.
% Keeps the correct scan time and most MR parameters.
% Removes the hospital name etc. See DINFOSTRIP for retained parameters.
% Uses a workaround to preserve RescaleSlope and RescaleIntercept.
%
% Tested only for MR on data from the Philips advanced viewing workstation
%
% dicoman
% dicoman(param, value, ...)
%   Parameter-value pair examples:
%    'Info',1   - generates summary information on screen
%    'Info',2   - as above but does not write out any files
%    'SeriesNumbers',[1401 1601]  - outputs only the listed Series (in
%                    this example, reconstructions 1 for scans 14 and 16).
%    'FileNameAdjust',1  - will prepend series (scan) number, slice and 
%                           dynamic number to filename if they exist. 
%    'SliceNumbers', [4]  - Only output specified SliceNumbers (Philips
%                           only) (e.g. slice 4)
%    'Dynamics',[7]       - Only output specified temporal positions
%                           (1-based) (e.g. dynamic 7)
%    'PatientName','newname'  - replaces PatientName (defaults to blank)
%
% Example:
%  dicoman('PatientName','volunteer3','Info',1,'FileNameAdjust',1)
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also DICOMANON DINFOSTRIP


nv = length(varargin) ;
FileNameAdjust = 0 ; % default 
Info = 0 ;
PatientName = '' ;

for ip = 1:2:nv
    switch varargin{ip}
        case {'Info','info'}
            Info = varargin{ip+1} ;
            if ~ismember(Info, [0 1 2])
                disp(['Info value should be 0, 1 or 2'])
            end
        case {'SeriesNumberVec' , 'SeriesNumber','SeriesNumbers'}
            SeriesNumberVec = varargin{ip+1} ;
        case 'FileNameAdjust'
            FileNameAdjust = varargin{ip+1} ;
            if ~ismember(FileNameAdjust, [0 1])
                disp(['FileNameAdjust value should be 1 or 0'])
            end
        case 'SliceNumbers'
            SliceNumbers = varargin{ip+1} ;
        case 'Dynamics'
            Dynamics = varargin{ip+1} ;
        case 'PatientName'
            PatientName = varargin{ip+1} ;
        otherwise
            disp(['Unknown input parameter: ',varargin{ip}])
    end
end

if ispref('dicoman','folder_name')
    folder_name = getpref('dicoman','folder_name') ;
else
    folder_name = [] ;
end

folder_name = uigetdir(folder_name,'Select DICOM input folder' ) ;
if isnumeric(folder_name) && folder_name ==0
    return
else
    setpref('dicoman','folder_name',folder_name)
    disp(['Input folder: ',folder_name])
end

if Info~=2
    if ispref('dicoman','out_folder_name')
        out_folder_name = getpref('dicoman','out_folder_name') ;
    else
        out_folder_name = [] ;
    end
    out_folder_name = uigetdir(out_folder_name , 'Select OUTPUT folder') ;
    if isnumeric(out_folder_name) && out_folder_name ==0
        return
    else
        setpref('dicoman','out_folder_name',out_folder_name)
        disp(['Output folder: ',out_folder_name])
    end
else
    SeriesNumberVec = -99 ;
end

dtop = dir(folder_name) ;

% parse for sub-folders one level down
isubf = 0 ;
d = dtop ;

for ident = 1:numel(d)
    if dtop(ident).isdir 
        if strcmp(dtop(ident).name,'.')==0 && ...
                strcmp(dtop(ident).name,'..')==0
            isubf = isubf + 1;
            subf{isubf} = dtop(ident).name ;
            %Produce directory structure for each subfolder, 
            % add sub-folder to name and append overall direc 
            dsubf = dir(fullfile(folder_name,subf{isubf})) ;
            for ident = 1:numel(dsubf)
                dsubf(ident).name = fullfile(subf{isubf}, dsubf(ident).name) ;
            end
            d = [ d ; dsubf ];
        end
    end
end


ikeep = [] ;

% record which dir entries are DICOM files
% (also captures the PS_* and XX_* files output by ViewForum)
for ident = 1:numel(d)
    if ~d(ident).isdir && d(ident).bytes > 128
        if isdicom(fullfile(folder_name,d(ident).name))
            ikeep = [ikeep ident] ;
        end
    end
end

d = d(ikeep) ;
nd = numel(d) ;

if exist('subf','var')
    nsubf = numel(subf) ;
else
    nsubf = 0 ;
end

hw = waitbar(0,['Processing ',num2str(nd),' files in ',...
    num2str(1+nsubf),' folders']) ;

studyUIDin = '' ;
seriesUIDin = '' ;

if Info
    sn = zeros([1 nd]) ;
    sl = zeros([1 nd]) ;
    tp = zeros([1 nd]) ;
end

tmpfn = tempname ;

for id = 1:nd
    waitbar(id/nd,hw) ;
    dinfo = dicominfo(fullfile(folder_name,d(id).name)) ;
    
    % insert skip if View Forum XX or PS files
    if strcmp(dinfo.SOPClassUID, '1.2.840.10008.5.1.4.1.1.11.1') == 0 && ...
            strcmp(dinfo.SOPClassUID, '1.2.840.10008.5.1.4.1.1.66') == 0
        
        if Info
            sn(id) = dinfo.SeriesNumber ;
            if isfield(dinfo,'Private_2001_100a')
                sl(id) = dinfo.Private_2001_100a ;
            else
                sl(id) = NaN ;
            end
            if isfield(dinfo,'TemporalPositionIdentifier')
                tp(id) = dinfo.TemporalPositionIdentifier ;
            else
                tp(id) = NaN ;
            end
        end
        
        [X, map] = dicomread(dinfo) ;
        
        if strcmp(studyUIDin, dinfo.StudyInstanceUID) == 0
            studyUID = dicomuid;
            seriesUID = dicomuid ;
            studyUIDin = dinfo.StudyInstanceUID ;
            seriesUIDin = dinfo.SeriesInstanceUID ;
        end
        
        % new series, update uid
        if strcmp(seriesUIDin, dinfo.SeriesInstanceUID) == 0
            seriesUID = dicomuid ;
            seriesUIDin = dinfo.SeriesInstanceUID ;
        end
        
        
        dinfored = dinfostrip(dinfo) ;
        dinfored.PatientName = PatientName ;
        dinfored.SeriesInstanceUID = seriesUID ;
        dinfored.StudyInstanceUID = studyUID ;
        
        % output
        if (exist('SeriesNumberVec','var') && ...
                ~ismember(dinfored.SeriesNumber, SeriesNumberVec)) || ...
            (exist('SliceNumbers','var') && isfield(dinfo,'Private_2001_100a') && ~ismember(dinfo.Private_2001_100a, SliceNumbers)) || ...
            (exist('Dynamics','var') && isfield(dinfo,'TemporalPositionIdentifier') && ~ismember(dinfo.TemporalPositionIdentifier, Dynamics))
            %do nothing because this file is not in the series to output
        else
            % subfolder names are in the file name. Create if they do not exist
            dicffn = fullfile(out_folder_name,d(id).name) ;
            % adjust output file name if requested
            if FileNameAdjust
                [pn,fn,ext] = fileparts(dicffn) ;
                fn = [fn ext] ;
                if isfield(dinfo,'TemporalPositionIdentifier')
                    fn = ['tp', num2str(dinfo.TemporalPositionIdentifier,'%03u'), fn] ;
                end
                if isfield(dinfo,'Private_2001_100a')
                    fn = [ 'sl', num2str(dinfo.Private_2001_100a,'%03u'), fn];
                end
                if isfield(dinfo,'SeriesNumber')
                    fn = [ 'sn', num2str(dinfo.SeriesNumber,'%04u'),fn] ;
                end
                dicffn = fullfile(pn,fn);
            end
            
            [dicdir] = fileparts(dicffn) ;
            if ~exist(dicdir,'dir')
                mkdir(dicdir) ;
            end
            
            % below is a work around as dicomwrite in create mode does not
            % seem to save RescaleIntercept
            
            status = dicomwrite(X, map, tmpfn, dinfored, ...
                'WritePrivate', true) ;
            
            tinfo = dicominfo(tmpfn) ;
            delete(tmpfn)
            
            if isfield(dinfored,'RescaleIntercept')
              tinfo.RescaleIntercept = dinfored.RescaleIntercept ;
            end
            if isfield(dinfored,'RescaleSlope')
              tinfo.RescaleSlope = dinfored.RescaleSlope ;
            end
            if isfield(dinfored,'RescaleType')
              tinfo.RescaleType = dinfored.RescaleType ;
            end
            
            status = dicomwrite(X, map, dicffn, tinfo, ...
                'CreateMode','Copy','WritePrivate', true) ;
        end
        
    end
    
end % for

if Info > 0
    firstpass = 1 ;
    [usn, md] = unique(sn) ;
    for isn = 1: numel(usn)
        if usn(isn) ~= 0
            loc = ismember(sn, usn(isn)) ;
            
            dinfo = dicominfo(fullfile(folder_name,d(md(isn)).name)) ;
            if firstpass
                if isfield(dinfo,'StudyDescription')
                    disp([dinfo.StudyDescription])
                end
                firstpass = 0 ;
            end
            if isfield(dinfo,'SeriesDescription')
                disp(['Series: ',num2str(usn(isn),'%4u'), '  ', ...
                    dinfo.SeriesDescription, ...
                    '  #files: ', num2str(sum(loc)), ])
            else
                disp(['Series: ',num2str(usn(isn),'%4u'), '  ', ...
                    '  #files: ', num2str(sum(loc)), ])
            end
            if ~isnan(max(sl(loc)))
                disp([' #SliceNumbers: ', num2str(length(unique(sl(loc))))])
            end
            if ~isnan(max(tp(loc)))
                disp([' #times: ', num2str(length(unique(tp(loc)))) ])
            end
            disp([' '])
        end
    end
end

close(hw)

    