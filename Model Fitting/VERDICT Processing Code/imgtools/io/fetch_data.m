function dats = fetch_data(dats)
% FETCH_DATA Fetch data from Zenodo or similar
%
%  dats = fetch_data(dats)
%
% dats is a structure with possible fields:
%  zenodoid     Zenodo id (required) e.g. '41056'
%  data_file    Name of data file
%  zenodotoken  Access token string required when data is closed access
%  local_dir    Local directory where file is saved. Opens uigetdir if not
%               present
%
%  local_file   Full filename of local file (currently ignored on input)
%
% Example usage:
%   dats.zenodoid = '400651'
%   dats.zenodotoken = get_token('zenodo:400651') ;
%   dats.data_file ='IM_0005' ;
% 
%   dats = fetch_data(dats) ;
%
%
% D.Atkinson@ucl.ac.uk
%
% See also TEMPDIR GET_TOKEN
%

if ~isfield(dats,'zenodoid')
    error(['Requires a Zenodo ID'])
else
    if isinteger(dats.zenodoid) % should be string 
        dats.zenodoid = num2str(dats.zenodoid) ;
    end
end

if ~isfield(dats,'local_dir') 
    dats.local_dir = '';
end
messg = 'Select folder for local file' ;
disp(messg)
dats.local_dir = pref_uigetdir('fetch_data', 'local_dir', ...
                    dats.local_dir, messg) ;


token_str = '';
if isfield(dats,'zenodotoken')
    token_str = ['?access_token=', dats.zenodotoken] ;
end

url_base = ['https://zenodo.org/api/deposit/depositions/', dats.zenodoid ] ;

zinfo = webread([url_base, token_str]) ;
disp([zinfo.metadata.title])
disp([zinfo.metadata.description])
disp(['Publication Date: ',zinfo.metadata.publication_date])
disp(['<a href="',zinfo.doi_url,'">',zinfo.doi_url,'</a>' ])
disp(['modified: ', zinfo.modified ])

zdata = webread([url_base, '/files', token_str]) ;


nf = length(zdata) ;

zfs = {zdata.filename} ;

[C, ia, ib] = intersect(zfs, dats.data_file) ;
if isempty(ia)
    error(['File not found on Zenodo.'])
end
if length(ia)>1
    warning(['Mulitple files with name: ', dats.data_file,'. Using first'])
    izfs = ia(1) ;
else
    izfs = ia ;
end


options = weboptions('RequestMethod','GET');

outfile1 = fullfile(dats.local_dir, zdata(izfs).filename) ;
outfile = websave(outfile1, [ zdata(izfs).links.download, token_str], options);
disp(['File saved to ',outfile])

dats.local_file = outfile ;




    