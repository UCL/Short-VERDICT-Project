function fns = dflist(folder)
% DFLIST List of DICOM files (recursive down folders)
%
% Example usage
% fns = dflist(folder)   ;  % returns a cell array of DICOM filenames
% fns = dflist ; % calls pref_uigetdir GUI
%

if nargin ==0 || isnumeric(folder) || ~exist(folder,'dir')
    folder = pref_uigetdir('dflist','dir',[],'Select upper folder for DICOMs') ;
    if isnumeric(folder)
        fns={}; 
        return
    end
end

d = dir(folder) ;

ndent = length(d) ;

fcount = 0 ;

for ident = 1:ndent
    if strcmp(d(ident).name,'.') == 1 ||  strcmp(d(ident).name,'..') == 1
        continue
    end
    
    if d(ident).isdir == true
        fsubd = dflist(fullfile(folder,d(ident).name)) ;
        nsub = length(fsubd) ;
        for isub = 1: nsub
            fcount = fcount + 1 ;
            fns{fcount} = fsubd{isub} ;
        end
        
    else
        % directory entry is a file
        if d(ident).bytes > 128
            ffile = fullfile(folder,d(ident).name) ;
            if isdicom(ffile)
                fcount = fcount + 1 ;
                fns{fcount} = ffile ;
            end
        end
    end
end
            
    
        