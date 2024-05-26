function a2p(fname)
% A2P Add to path only if folder present
%
% Only adds to path if present
if exist(fname,'dir')
    addpath(fname) ;
end