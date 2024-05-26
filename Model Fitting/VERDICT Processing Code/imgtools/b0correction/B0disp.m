function B0disp(vB0)
% B0disp

[ny,nx, nz] = size(vB0) ;

vrs = reshape(vB0,[ny nx 1 nz]) ;

figure
montage(vrs, 'DisplayRange', [-100 100]), colorbar

% Compute spherical harmonics and subtract/add to this B0 to understand
% issues. 
% Initially, take isocentre to be at DICOM origin,
% Z to be FH, 
%
%
% From Schar, code is:
%  delta B0 = offset + G_x.x + ...
%
% Coordinates. Here require the x,y,z coordinate of each pixel in the B0
% map. Will just use LPH for now
%
% Convert to Hz using deltaTE and gamma.
%


