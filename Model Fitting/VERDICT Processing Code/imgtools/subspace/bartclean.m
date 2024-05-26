function bartclean(varargin)
% BARTCLEAN Clean folder of BART *.cfl and corresponding hdr files
% Default behavior is to delete without confirmaton all cfl and corresponding
% hdr files in the current folder.
%
% bartclean
% bartclean('dryrun',true)  Note: default is to delete
%
% David Atkinson, University College London
%

dryrun = false ;
for ipv = 1:2:length(varargin)
    param = varargin{ipv} ;
    val   = varargin{ipv+1} ;
    switch param
        case 'dryrun'
            dryrun = val ;
        otherwise
            error(['Unknown parameter: ',param])
    end
end

% We pick hdr files to be only those that correspond to a cfl so as not to
% remove any NIfTI ones by accident.

flist = dir('*.cfl') ;

itodelete = 0 ; % running total of number of files to be deleted

for ifile = 1:length(flist)
    cflfn = flist(ifile).name ;
    
    [pn, fn, ext] = fileparts(cflfn) ;
    
    hdrfn = fullfile(pn, [fn '.hdr']) ;
    
    if exist(hdrfn,'file')
        itodelete = itodelete + 1 ;
        deletelist{itodelete} = hdrfn ;
    else
        warning(['No corresponding hdr file to: ',cflfn])
    end
    
    itodelete = itodelete + 1 ;
    deletelist{itodelete} = cflfn ;
end


disp([num2str(itodelete),' files to be deleted.'])

% Delete done in separate section so that a dryrun can be performed
for idel = 1: itodelete
    if ~dryrun
        delete(deletelist{idel} );
    else
        disp([deletelist{idel}, ' is in delete list.'])
    end
end

