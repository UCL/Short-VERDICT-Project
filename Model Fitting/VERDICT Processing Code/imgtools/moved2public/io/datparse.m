function dinfo = datparse(datfn)
% DATPARSE Parse data files; Philips PAR files and DICOM
%  dinfo = datparse
%  dinfo = datparse(datfn) 
%  dinfo = datparse(dselect) Uses dselect GUI for MultiFrame selection
%
%  Returns dinfo empty if there is an error parsing file.
%
% Intended to be used with d2mat. Calls 
%   parparse for Philips PAR/REC, otherwise
%   dmfparse for multi-frame DICOM 
%     Multiple selections allowed for MF DICOM series that are 
%     similar e.g. multi flip angle.
%
% For multi-flip angle in SingleFrame DICOM, place in one folder, or e.g. 
%   MFA upper folder with subfolders FA2, FA11 etc
%
% Currently does not call xmlparse. dmfparse will subsequently call 
% dparse (non multiframe reader to set folder) if it detects a 
% non multiframe DICOM file.
%
% Copyright, 2019, David Atkinson  
% D.Atkinson@ucl.ac.uk
%
% See also DSELECT DPARSE D2MAT XMLPARSE DMFPARSE PARPARSE 

% The logic for multi-selection here is cumbersome and not elegant.
% Needs a re-write!

%  Note, dselect now returns a cell array of file names
if exist('datfn','var') 
    if iscell(datfn)
        datfn_test = datfn{1} ;
    else
        datfn_test = datfn ;
    end
else
    datfn_test = 'XX' ;
end

if nargin==0 || ~exist(datfn_test,'file')
    if ispref('datparse','datfn')
        datfn = getpref('datparse','datfn') ;
    else
        datfn = [] ;
    end
                        
   [fn,pn] = uigetfile({'*'}, ...
       'Select 1 PAR or 1 REC or DICOM (1 example single frame or 1+ multi)', datfn, ...
       'MultiSelect','on') ;
   if isequal(fn, 0) ; dinfo = [] ; return; end ;
   
   if ~iscell(fn)
     datfn = fullfile(pn,fn) ;
     setpref('datparse','datfn',datfn)
   else
       clear datfn ;
       for ifile = 1:length(fn)
           datfn{ifile} = fullfile(pn,fn{ifile}) ;
       end
       setpref('datparse','datfn',datfn{1})
   end
end

if ~iscell(datfn)
    disp(['Parsing file: ',datfn])
    
    [pathstr, name, ext] = fileparts(datfn) ;
    
    switch ext
        case {'.PAR','.par'}
            dinfo = parparse(datfn) ;
        case {'.REC'}
            datfn = fullfile(pathstr,[name,'.PAR']) ;
            dinfo = parparse(datfn) ;
            DATFN = datfn ;
        case {'.rec'}
            datfn = fullfile(pathstr,[name,'.par']) ;
            dinfo = parparse(datfn) ;
            DATFN = datfn ;
        otherwise
            dinfo = dmfparse(datfn) ;
    end
    
else
    dinfo = dmfparse(datfn) ;
end
