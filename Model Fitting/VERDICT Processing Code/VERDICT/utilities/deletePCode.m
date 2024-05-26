function deletePCode
%deletePCode Deletes p files in psource 


project = matlab.project.currentProject;

delete(fullfile(project.RootFolder,'psource','*.p'))


end