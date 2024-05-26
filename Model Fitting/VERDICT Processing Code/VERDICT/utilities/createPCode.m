function createPCode
%createPCode Creates p file in psource from source
% 

project = matlab.project.currentProject;
currd = pwd ;

cd(fullfile(project.RootFolder,'psource'))

pcode(fullfile(project.RootFolder,'source','*.m'))

cd(currd)

end