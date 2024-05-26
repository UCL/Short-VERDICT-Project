function backup_check
% BACKUP_CHECK Check backup (compares folders)
%
% Checks that files in one folder (orig) and its sub-folders, are in 
%  folder new or its sub-folders.
% Checks file names and file sizes.
% Assumes no duplicate filenames.
%
% David Atkinson
%

disp('Select orig folder')
folder_orig = pref_uigetdir('backup_check','dorig') ;
disp('Select new folder')
folder_new  = pref_uigetdir('backup_check','dnew') ;

dorig = dir([folder_orig, filesep, '**', filesep, '*']) ;
dnew  = dir([folder_new, filesep, '**', filesep, '*']) ;

norig = length(dorig) ;

fnew = {dnew.name} ;

hw = waitbar(0,[num2str(norig),' files in: ',folder_orig]) ;

for idorig = 1: norig
    waitbar(idorig/norig, hw) ;

    if dorig(idorig).isdir
        continue
    end
    ct = contains(fnew, dorig(idorig).name) ;
    switch sum(ct)
        case 0
            disp(['NOT PRESENT: ',dorig(idorig).name]) 
        case 1
            % Exactly 1 name match, now check size
            if abs(dorig(idorig).bytes - dnew(ct).bytes) > 1
                disp(['SIZE MISMATCH: ', ...
                    dorig(idorig).name,' (',num2str(dorig(idorig).bytes),' orig) ', ...
                    ' (',num2str(dnew(ct).bytes),') new'])
            end
        otherwise
            disp(['MULTIPLE PRESENT: ',dorig(idorig).name]) 
    end
end

close(hw)

disp(['Analysed ',num2str(norig),' entries in ',folder_orig])
