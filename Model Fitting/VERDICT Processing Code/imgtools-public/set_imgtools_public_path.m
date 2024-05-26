function set_imgtools_public_path(dirin)
% SET_IMGTOOLS_PUBLIC_PATH Add Git folders to MATLAB path
% 
% The imgtools-public repository contains public versions of code developed
% largely at UCL.
%
% Assuming that you have a copy/clone of the full repository.
% Type 'edit startup', add these lines to startup.m (change the 1st 
% to point to the correct place) and restart MATLAB:
%   
%   gitdir  = 'C:\home\matlab\imgtools-public' ; % local copy
%   addpath( gitdir )                     % adds the path to this function 
%   set_imgtools_public_path( gitdir )    % places files and folders on path
% 
% Third party code is no longer in the imgtools repository (or is in the
% process of being moved out). Downloaded code needs to be placed in a
% folder, for example called 'external' and this folder placed on the
% matlab path.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

% a2p( fullfile( dirin, 'ASL')) ;
% a2p( fullfile( dirin, 'b0correction')) ;
% a2p( fullfile( dirin, 'b1mapping')) ;

% a2p( fullfile( dirin, 'dual_echo')) ;
% a2p( fullfile( dirin, 'DSC')) ;
% a2p( fullfile( dirin, 'ISMRMRD')) ; % see also external\ISMRMRD
a2p( fullfile( dirin, 'education')) ;
a2p( fullfile( dirin, 'external')) ;
a2p( fullfile( dirin, 'external/SpinCalc1p3')) ;
a2p( fullfile( dirin, 'io')) ;
% 
% a2p( fullfile( dirin, 'diffusion')) ; % some functions need NODDI
% %   http://cmic.cs.ucl.ac.uk/mig/index.php?n=Tutorial.NODDImatlab
% a2p( fullfile( dirin, 'education')) ;
a2p( fullfile( dirin, 'GUI')) ;
% a2p( fullfile( dirin, 'SWI')) ;
% a2p( fullfile( dirin, 't1mapping')) ;
a2p( fullfile( dirin, 'tools')) ;

% 
% a2p( fullfile( dirin, 'PC')) ;


function a2p(fname)
% A2P Add to path only if folder present
%
% Only adds to path if present
if exist(fname,'dir')
    addpath(fname) ;
end
        
