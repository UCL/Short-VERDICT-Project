function dsfsort(varargin)
% DSFSORT Dicom Single Frame Sort into folders
% Moves Single Frame DICOM files into sub-folders named after
% the scan number and protocol name. Files are moved but not renamed.
% Recommend working on a copy of originals. DICOMDIR file is not updated.
%
% All single frame DICOMS must be in one folder (does not look in
% sub-folders). New sub-folders are placed below a new folder with default 
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

dirlist = dir(dicom_folder) ;

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

for id = 1: length(dirlist)
    if ~dirlist(id).isdir % not a folder
        if dirlist(id).bytes >= 132 % min size for a DICOM
            fn = fullfile(dicom_folder,dirlist(id).name) ;
            if isdicom(fn)
                dinfo = dicominfo(fn) ;
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
                        disp([num2str(round(id/length(dirlist)*100)), ...
                            '% complete, made folder: ',series_folder])
                    end
                end
                
                [status, mess, messid] = movefile(fn, series_folder) ;
                if status == 0
                    disp([mess,'  ',messid])
                end
            end
        end
    end
end

            
        