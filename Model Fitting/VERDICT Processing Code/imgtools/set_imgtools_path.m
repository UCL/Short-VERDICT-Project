function set_imgtools_path(dirin)
% SET_IMGTOOLS_PATH Add Git folders to MATLAB path
% 
% The imgtools files are developed in a GIT repository. A restricted set 
% can be exported to form the more public imgtools zip archive. 
%
% If your local folder, for example here called 'C:\home\matlab\imgtools' ,  
% contains either your local git repository, or the unzipped files, you 
% need to add it to your MATLAB path. 
%
% For users of the full repository:
% Type 'edit startup', add these lines (change 1st if necessary), and 
% restart MATLAB:
%   
%   gitdir  = 'C:\home\matlab\imgtools' ; % local working copy
%   addpath( gitdir )                     % adds the path to this function 
%   set_imgtools_path( gitdir )           % places files and folders on path
%
% This set_imgtools_path.m file is not exported, users of the zip file should see the
% instructions for modifying the MATLAB path in the README.txt file.
% 
% Developers should keep the README.txt file up to date. 
% To exclude files from the exported zip archive, edit .gitattributes, 
% commit and then export.
% 
% Third party code is no longer in the imgtools repository (or is in the
% process of being moved out). Downloaded code needs to be placed in a
% folder, for example called 'external' and this folder placed on the
% matlab path.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

a2p( fullfile( dirin, 'ASL')) ;
a2p( fullfile( dirin, 'b0correction')) ;
a2p( fullfile( dirin, 'b1mapping')) ;
a2p( fullfile( dirin, 'CEST')) ; % CEST, currently not exported
a2p( fullfile( dirin, 'dual_echo')) ;
a2p( fullfile( dirin, 'DSC')) ;
a2p( fullfile( dirin, 'ISMRMRD')) ; % see also external\ISMRMRD
a2p( fullfile( dirin, 'io')) ;

a2p( fullfile( dirin, 'diffusion')) ; % some functions need NODDI
%   http://cmic.cs.ucl.ac.uk/mig/index.php?n=Tutorial.NODDImatlab
a2p( fullfile( dirin, 'education')) ;
a2p( fullfile( dirin, 'GUI')) ;
a2p( fullfile( dirin, 'SWI')) ;
a2p( fullfile( dirin, 'subspace')) ;
a2p( fullfile( dirin, 't1mapping')) ;
a2p( fullfile( dirin, 'tools')) ;
a2p( fullfile( dirin, 'ml')) ;
a2p( fullfile( dirin, 'motion')) ;
a2p( fullfile( dirin, 'motion', 'autofocus')) ;
a2p( fullfile( dirin, 'motion', 'RDDR')) ;
% Folders below are with-held from export (in GIT, edit .gitattributes )
a2p( fullfile( dirin, 'motion', 'OPT_FLOW')) ; 

a2p( fullfile( dirin, 'PC')) ;
a2p( fullfile( dirin, 'Philips')) ; %excluded from export
a2p( fullfile( dirin, 'UNDISTORT')) ;

function a2p(fname)
% A2P Add to path only if folder present
%
% Only adds to path if present
if exist(fname,'dir')
    addpath(fname) ;
end
        
