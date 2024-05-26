function startup_verdict_project
% STARTUP_PROJECT_VERDICT Adds SPAMS to path and environment variables
% The top folder needs to already be on the MATLAB path so that this
% function can find it.
%
%
% 

s = what('spams-matlab-v2.6') ; % top folder

if ~isempty(s)
    addpath(fullfile(s.path,'')) ; 
    addpath(fullfile(s.path,'test_release'));
    addpath(fullfile(s.path,'src_release'));
    addpath(fullfile(s.path,'build'));
    setenv('MKL_NUM_THREADS','1')
    setenv('MKL_SERIAL','YES')
    setenv('MKL_DYNAMIC','NO')
end
