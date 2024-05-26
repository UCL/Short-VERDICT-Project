function setT1NIFTI(newT1, ffn)
% setT1NIFTI Sets the values in a T1 map to a user-specifiec constant value
% 
%  setT1NIFTI(newT1_ms)
%  setT1NIFTI(newT1_ms, ffn)
%
%    newT1_ms  is the new T1 value in ms
%    ffn       is the full file name of the input file (if absent, will
%               present a window to pick a file)
%
% Output file has all values set to newT1_ms (written inside file in units of
% seconds). Filename will be the original, with the newT1_ms value
% appended, e.g. 'H66.nii' would become 'H66_1500.nii'. 
% When creating the new name, the value of newT1 is rounded to the nearest integer.
% Any existing file with the same name will be overwritten.
%
% Copyright 2021. David Atkinson, University College London
% D.Atkinson@ucl.ac.uk
%
% See also pref_uigetfile
%

% Reads in original T1 map (NIfTI file)
% Set all values to user specified value newT1_ms
% Creates new file name.
% Saves new NifTI in same folder
%


if newT1 < 10
    warning('Low T1 entered, expect number in ms')
end

if nargin < 2 || ~exist(ffn,'file')
   % User select a file
   ffn = pref_uigetfile('setT1NIFTI','ffn') ;
end

if ~exist(ffn,'file')
    error(['File not found: ',ffn])
end

ninfo = niftiinfo(ffn) ;
V = niftiread(ninfo) ;

% reset values
newV = newT1/1000 * ones(size(V),'like',V) ;

% create new filename
[fpath,fname,fext] = fileparts(ffn) ;
newfname = [fname,'_', sprintf('%04u', round(newT1)) ] ;
disp(['New filename is: ',newfname])

newffn = fullfile(fpath, [newfname fext]) ;

niftiwrite(newV, newffn, ninfo) 


