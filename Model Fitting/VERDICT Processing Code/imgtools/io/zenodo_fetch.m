function filenms = zenodo_fetch(varargin)
% ZENODO_FETCH Fetch data from Zenodo site. Place in local file(s)
%
%  filenms = fetch_data(param, value, ...)
%
%  filenms is cell array of local file names
%
%  param can be:
%
%   'zenodoid' Zenodo ID (required) e.g. '41056'
%   'files2fetch' Cell array of files to fetch. Default is all.
%   'zenodotoken' Access token string required when data is closed access
%                 e.g. get_token('zenodo:400651') ;
%   'local_dir' Local directory where file(s) saved. Opens uigetdir if not
%               present. For temporary, use tempdir (default).
!! put in method to just get as below because cannot get API to work at the moment

% websave('tempnema','https://zenodo.org/record/1304454/files/NEMA_IQ.zip')
%
% Example 1. Closed Zenodo site with token retrieved from user's own
% get_token functon
%
% tok = get_token('zenodo:400651');
% filnms = zenodo_fetch('zenodoid','400651','zenodotoken',tok, ...
%                       'files2fetch',{'IM_0009', 'IM_0006'})
%
% Example 2. Open Zenodo site
% 
% filenms = zenodo_fetch('zenodoid','1304454' )
%
% Copyright, 2019, David Atkinson 
% D.Atkinson@ucl.ac.uk
%
% See also TEMPDIR GET_TOKEN
%

% defaults
local_dir = tempdir ;

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    switch varargin{ipv}
        case 'zenodoid'
            zenodoid = val ;
            if isinteger(zenodoid) % should be string 
              zenodoid = num2str(zenodoid) ;
            end
        case 'files2fetch'
            files2fetch = val ;        
        case 'zenodotoken'
            zenodotoken = val ;
        case 'local_dir' 
            local_dir = val ;
        otherwise
            error(['Unknown parameter: ',varargin{ipv}])
    end
end
    
if ~exist('zenodoid','var')    
    error('Requires a Zenodo ID')
end

messg = 'Select local folder for file(s)' ;
local_dir = pref_uigetdir('zenodo_fetch', 'local_dir', local_dir, messg) ;


if exist('zenodotoken','var')
   token_str = ['?access_token=', zenodotoken] ;
else
   token_str = '';
end

url_base = ['https://zenodo.org/api/deposit/depositions/', zenodoid ] ;

zinfo = webread([url_base, token_str]) ;
disp(' ')
disp([zinfo.metadata.title])
disp([zinfo.metadata.description])
disp(['Publication Date: ',zinfo.metadata.publication_date])
disp(['<a href="',zinfo.doi_url,'">',zinfo.doi_url,'</a>' ])
disp(['modified: ', zinfo.modified ])

zfs = {zinfo.files.filename} ;

options = weboptions('RequestMethod','GET');

if ~exist('files2fetch','var')
    files2fetch = zfs ;
end
if ~iscellstr(files2fetch)
    files2fetch = cellstr(files2fetch) ;
end

filenms = {} ;

for if2f = 1: length(files2fetch)
    do_download = true ;
    
    f2ffn = files2fetch{if2f} ;
    [~, ia, ~] = intersect(zfs, f2ffn ) ;
    
    if isempty(ia)
        warning(['File ',f2ffn,' not found on Zenodo.'])
        do_download = false ;
    end
    
    if length(ia)>1
        warning(['Mulitple files with name: ', f2ffn,'. Using first'])
        izfs = ia(1) ;
    else
        izfs = ia ;
    end

    
    outfile1 = fullfile(local_dir, f2ffn) ;
   
    % could add here code to check if file already downloaded. Simple to
    % check file exists, but more complicated to check mod date etc.
    
    if do_download
      outfile = websave(outfile1, [ zinfo.files(izfs).links.download, token_str], options);
      disp(['File saved to ',outfile])
      filenms{if2f} = outfile ;
    end
    
end




    