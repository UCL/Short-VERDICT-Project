function folder_name = pref_uigetdir(group, pref, user_folder, dlg_title)
% PREF_UIGETDIR Calls uigetdir with stored user preferences
% Stores folders so that subsequent calls to this function place the user
% in the same folder as before.
%
% folder_name = pref_uigetdir(group, pref)
% folder_name = pref_uigetdir(group, pref, user_folder)
% folder_name = pref_uigetdir(group, pref, user_folder, dlg_title)
%
% If user_folder exists, it will  be passed as the output and set to the
% stored folder name. If not, uigetdir will be called.
%
% David Atkinson   D.Atkinson@ucl.ac.uk
% See also UIGETDIR GETPREF SETPREF
%
if nargin > 2 && exist(user_folder,'dir')
    folder_name = user_folder ;
else
    
    if ispref(group,pref)
        defdir = getpref(group, pref);
    else
        defdir = [] ;
    end
    
    if nargin < 4
        dlg_title = 'Select Folder' ;
    end
    if ismac
        disp([dlg_title])
    end
    [folder_name] = uigetdir(defdir,dlg_title) ;
    
    
    if folder_name == 0
        warning(['No folder specified'])
        return
    end
    
end
setpref(group, pref, folder_name)

