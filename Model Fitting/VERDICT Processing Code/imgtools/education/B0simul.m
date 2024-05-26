function B0warping_simul
% B0warping_simul Simulation of B0 warping and correction
%  This just applies shifts that are proportional to field offset, no FFTs
%  All images here are of the same size and deal only with intrinsic
%  coordinates. Demonstrates use of scatteredInterpolant and imwarp
%
%  B0warping_simul
%
% Copyright 2020, David Atkinson
% D.Atkinson@ucl.ac.uk
%

% Extension to real data needs:
%  Direction and row/column for pixel/Hx - see dwfs
%  Arbitrary pixel spacings, IOP and IPP for the two datasets.
%  Replace imwarp with tformarray if need to handle complex dimensions.
%

N = 128 ; % grid size. Currently DWI and B0map are all same sixe here

map_pH = [0.1 0] ; % Pixel per Hz  [x y] (Reciprocal of Hz/pixel)
dwi_pH = [0   1]  ; 

B0true = 3*peaks(N) ; % Ground truth B0 field
dwitrue = phantom(N) ; % Ground truth DWI-EPI image
[Xt, Yt] = meshgrid(1:N, 1:N) ; % x and y coordinates of regular grid

% B0map scan is itself distorted
Xm = Xt + B0true*map_pH(1) ; % Location in B0 map scan where true B0 maps
Ym = Yt + B0true*map_pH(2) ;

Xd = Xt + B0true*dwi_pH(1) ; % Location in DWI scan where points map to.
Yd = Yt + B0true*dwi_pH(2) ;

% Compute actually measured (lightly distorted B0map)
Fm = scatteredInterpolant(Xm(:), Ym(:), B0true(:)) ; 
B0map = Fm(Xt, Yt) ;

% Compute distorted DWI image, based onground truth
Fd = scatteredInterpolant(Xd(:), Yd(:), dwitrue(:)) ; 
dwi = Fd(Xt, Yt) ;


% Now perform unwarping without using ground truth.

% Find location in 'true' space where B0map grid points are
Xti = Xt - B0map*map_pH(1) ; % true irregular
Yti = Yt - B0map*map_pH(2) ;

Fti = scatteredInterpolant(Xti(:), Yti(:), B0map(:) ) ;

% B0map warped to space of undistorted and interpolated at regular grid
B0mapt_sim = Fti(Xt, Yt) ; % B0map at true points from measured B0map

% Calculate displacement field for true to DWI. 
D = zeros([size(dwi,1) size(dwi,2) 2]) ;
D(:,:,1) = B0mapt_sim*dwi_pH(1) ; % D maps from regular B0map to distorted DWI
D(:,:,2) = B0mapt_sim*dwi_pH(2) ;

dwiti = imwarp(dwi,D) ; % analagous to D mapping fixed to moving, 
% here the dwi is 'moving' and we get back to fixed 'true' frame


eshow(dwiti)
eshow(dwiti-dwitrue)
eshow(B0mapt_sim)
eshow(B0mapt_sim - B0true)
eshow(B0true)
eshow(B0map) 
eshow(dwi)
eshow(dwitrue)
