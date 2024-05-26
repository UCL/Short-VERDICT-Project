function [Xs, Ys, X, Y] = B0shift( B0_Hz, towarp, IOP, towarp_wfs_hzpp, towarp_wfs_dir )
%B0shift Pixel shift due to B0.
%   Shifted pixel coordinates due to B0 off-resonance.
%
% [Xs, Ys] = B0shift( B0_Hz, towarp, IOP, towarp_wfs, towarp_wfs_dir )
%
% B0_Hz is sampled at points in towarp
% IOP, fat_shift_dir_lph and PixelBandwidth relate to the towarp image
% Xs, Ys are shifted column and row pixel coordinates
%  (unshifted, centre of top left pixel is (1,1))
%
% 2D shifts only, although all slices of an input volume will be processed.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

% check sizes
if ~isequal(size(B0_Hz), size(towarp))
    warning(['Unequal input sizes: '])
end

% Displacements 
shifts = double(-B0_Hz / towarp_wfs_hzpp) ;

% Regular grid coordinates, TLC = (1,1)
xgv = 1:size(towarp,2) ;
ygv = 1:size(towarp,1) ;
[X,Y] = meshgrid(xgv,ygv) ;

% Scalar dot product of fat shift direction with IOP to determine shift in
% this image.

wfs_row = dot(towarp_wfs_dir, IOP(1:3)) ;
wfs_col = dot(towarp_wfs_dir, IOP(4:6)) ;
disp(['WFS row: ',num2str(wfs_row), ', col: ',num2str(wfs_col)])

nslice = size(towarp,3) ;
Xs = zeros([size(towarp,1) size(towarp,2) nslice]) ;
Ys = Xs ;

X = repmat(X, [1 1 nslice]) ;
Y = repmat(Y, [1 1 nslice]) ;

if abs(wfs_row) > abs(wfs_col)
    if wfs_row > 0
        Xs = X + shifts ;
    else
        Xs = X - shifts ;
    end
    Ys = Y ;
else
    if wfs_col > 0
        Ys = Y + shifts ;
    else
        Ys = Y - shifts ;
    end
    Xs = X ;
end
    



