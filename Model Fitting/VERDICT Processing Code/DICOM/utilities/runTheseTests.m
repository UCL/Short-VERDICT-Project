function runTheseTests()
%RUNTHESETESTS Run all tests in the project.
%   Taken from MathWorks example

project = matlab.project.currentProject;
runtests(project.RootFolder);

end