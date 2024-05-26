function dsfsort2(varargin)
% DSFSORT2 Dicom Single Frame Sort into folders
% Moves Single Frame DICOM files into sub-folders named after
% the scan number and protocol name. Files are moved but not renamed unless
% the name would be duplicated in that folder - in that case it is given a
% unique filename.
% Recommend working on a copy of originals. DICOMDIR file is not updated.
%
% Single frame DICOMS can be in sub-folders. New
% sub-folders are placed below a new folder with default
% name 'sorted' which is placed in the same folder as the original DICOMs.
%
% Files that are not DICOM are not moved.
%
% Philips XX files associated with a series are also moved.
% The ExamCard XX file goes into a folder with name 0000_ExamCard.
%
% For example, original structure:
% DICOM
%   IM01  (scan 1)
%   IM02  (scan 2)
%   IM03  (scan 2)
%
% would become:
%
% DICOM
%  sorted
%    0101_survey    0201_T2W
%       IM01          IM02
%                     IM03
%
% Intended to cope with large numbers of files from Ingenia
%
% Example Use
% % Original folder structure is changed and small risk of data overwrite -
% % recommend to make a copy of the original data first.
%
% dsfsort
% dsfsort(param, val, ...)
%   parameters
%     'dicom_folder' Specify location of DICOM files, default is '',
%                     if folder does not exist, function calls uigetdir.
%     'sort_folder'  Location of sorted files relative to dicom_folder,
%                     default is 'sorted'.
%
%
% D.Atkinson@ucl.ac.uk
%
% See also DSFSORT DFLIST
%

% defaults
dicom_folder = '' ;      % location of Single Frame DICOMs
sort_folder = 'sorted' ; % this is relative to dicom_folder

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    
    switch varargin{ipv}
        case 'sort_folder'
            sort_folder = val ;
        case 'dicom_folder'
            dicom_folder = val ;
        otherwise
            error(['Unknown parameter: ',varargin{ipv}])
    end
    
end

if ~exist(dicom_folder,'dir')
    disp(['Select folder of Single Frame DICOMS'])
    dicom_folder = uigetdir('','Select folder of Single Frame DICOMs') ;
else
    disp(['Using dicom_folder: ',dicom_folder])
end

fnlist = dflist(dicom_folder) ;
% dirlist = dir(dicom_folder) ;

% Check if folder already exists
sort_folder = fullfile(dicom_folder, sort_folder) ;
if exist(sort_folder,'dir')
    error(['sort_folder ',sort_folder,' already exists.'])
else
    [status, mess, messid] =  mkdir(sort_folder) ;
    if status == 0
        warning(['Failed to create folder: ',sort_folder,'. ',mess,' ',messid])
    elseif ~isempty(mess)
        disp(mess)
    else
        disp(['Made folder: ',sort_folder])
    end
end

for id = 1: length(fnlist)
    ffn = fnlist{id} ;
    
    dinfo = dicominfo(ffn) ;
    if isfield(dinfo,'SeriesNumber')
        sn = num2str(dinfo.SeriesNumber,'%04u') ;
    else
        sn = 'no_series' ;
    end
    if isfield(dinfo,'ProtocolName')
        % tidy up characters that cannot go in filename
        protnm = dinfo.ProtocolName ;
        k = strfind(protnm,' ') ;
        protnm(k) = '_' ;
        k = strfind(protnm,'/') ;
        protnm(k) = '_' ;
        k = strfind(protnm,'\') ;
        protnm(k) = '_' ;
        fnprot = ['_',protnm];
    else
        protnm = '';
        fnprot = '';
    end
    series_folder = fullfile(sort_folder,[sn,fnprot]) ;
    
    if ~exist(series_folder,'dir')
        [status, mess, messid] = mkdir(series_folder) ;
        if status == 0
            error(['Failed to mkdir: ',series_folder])
        else
            disp([num2str(round(id/length(fnlist)*100)), ...
                '% complete, made folder: ',series_folder])
        end
    end
    
    [pn,fn,ext] = fileparts(ffn) ;
    destfn = fullfile(series_folder,[fn ext]) ;
    
    % If destination file exists, use a unique filename
    if exist(destfn,'file')
        tffn = tempname ;
        [pn,tfn,tfext] = fileparts(tffn) ;
        destfn = fullfile(series_folder,[tfn tfext]) ;
    end
    
    [status, mess, messid] = movefile(ffn, destfn) ;
    if status == 0
        disp([mess,'  ',messid])
    end
    
end


