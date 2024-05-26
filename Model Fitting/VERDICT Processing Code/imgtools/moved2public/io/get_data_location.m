function prefval = get_data_location(prefname, dtype, varargin)
% GET_DATA_LOCATION Provide location of data for testing etc
% Data locations stored in preference variables. Default behaviour is to
% only call relevant GUI (uigetfile, uigetdir or dselect) when needed.
%
% folder = get_data_location(prefname, 'dir', Name, Value, ...)
% file   = get_data_location(prefname, 'file', Name, Value, ...)
% files  = get_data_location(prefname, 'files', Name, Value, ...)
% files  = get_data_location(prefname, 'dselect', Name, Value, ...)
%
% prefname  preference name where selection is stored (must follow 
%           Variable Naming rules - cannot start with a number)
%
% dtype 
%  'dir', 'file', 'files', 'dselect'
%
% If dselect GUI opens, it will use its own default folder.
%
%
% Name, Value pairs
%
% 'filter' for file(s), defaults to '*'
% 'dselectpvp' parameter, value pairs for passing to dselect
%
% 'gui'
%   'always' | {'ifneeded'} | 'never'
%
% 'store'
%   {'true'} | 'false'
%
%
% Example Usage:
%   folder = get_data_location('geom20191225', 'dir') ;
%   
%   filenm = get_data_location('geom20200122coronal', 'file', 'filter', ...
%                              '*.dcm')
%
%   cell_fns = get_data_location('geom20200122coronal', 'files', 'filter', ...
%                              '*.dcm')
%
%   fns = get_data_location('geom20200122coronal', 'dselect', ...
%             'dselectpvp',  {'message', 'Select T2 EMR and XX files'} )
%
%
% Example within a test fixture:
%  import matlab.unittest.fixtures.PathFixture
%  dataFolder = get_data_location('DICOM_20191211geometryIngenia','dir') ;
%  f = testCase.applyFixture(PathFixture(dataFolder));
%
%
% Copyright 2020, David Atkinson
% D.Atkinson@ucl.ac.uk
%
% See also DSELECT UIGETFILE UIGETDIR

gui = 'ifneeded' ; % default is that GUI only opens if no valid preference
store = true ; % store preference
filter = '*' ; % file filter
dselectpvp = {} ; % Parameter, value pairs passed to dselect

groupname = 'get_data_location' ; 

for ipv = 1:2:length(varargin)
    param = varargin{ipv} ;
    val = varargin{ipv+1} ;
    switch param
        case {'store'}
            store = val ;
        case 'gui'
            gui = val ;
        case 'filter'
            filter = val ;
        case 'dtype'
            dtype = val ;
        case 'dselectpvp'
            dselectpvp = val ;
        otherwise
            error(['Unknown param: ', param])
    end
end

validpref = false ;
prefval = [] ;
if ispref(groupname, prefname)
    prefval = getpref(groupname, prefname) ;
    switch dtype
        case 'dir'
            if isfolder(prefval)
                validpref = true ;
            end
        case 'file'
            if isfile(prefval)   % Actualy works for file or files
                validpref = true ;
            end
            MS = 'off' ;
        case 'files'
            MS = 'on' ;
            isf = isfile(prefval) ;
            if all(isf) 
                validpref = true ;
            end
        case 'dselect'
            if isfile(prefval)
                validpref = true ;
            end
        otherwise
            error('Unknown dtype')
    end
end
 
% GUI
if strcmp(gui,'never')
    if validpref==false
        warning('No valid preference and GUI set to never')
    end
elseif strcmp(gui,'always') || (strcmp(gui,'ifneeded') && validpref==false)
    switch dtype
        case 'dir'
            prefval = uigetdir(prefval) ;
            if ischar(prefval) && isfolder(prefval)
                validpref = true ;
            else
                validpref = false ;
            end
        
        case {'file', 'files'}            
            [fn, pn] = uigetfile(filter, [], prefval, 'MultiSelect',MS) ;
            ffn = fullfile(pn, fn) ;
            isf = isfile(ffn) ;
            if all(isf) 
                prefval = ffn ;
                validpref = true ;
            else
                validpref = false ;
            end
        case 'dselect'
            ffn = dselect(dselectpvp{:}) ;
            if isfile(ffn)
                prefval = ffn ;
                validpref = true ;
            else
                validpref = false ;
            end
    end
end

% Storage
if store && validpref==true
    setpref(groupname, prefname, prefval) ;
end



    

