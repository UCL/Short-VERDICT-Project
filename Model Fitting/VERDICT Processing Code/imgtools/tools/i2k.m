function ckdat = i2k(cimdat)
%I2K      Returns the complex k-space data, given complex image domain.
%         ckdat = i2k(cimdat) 
%
%         Handles 2D or 3D
%
%         Currently uses matlab FFT, may be modified in future to 
%         go either way to maintain compatibility with UCL and UMDS
%         code
%
%         I -> K FFT:   1 \sum exp(-)
%         K -> I IFFT:  1/N \sum exp(+)
%         David Atkinson , UMDS  
%
% @(#)i2k.m	1.5 , created 02/22/02
%
% See also H2K H2I I2H K2I K2H I2K1

ckdat = fftshift(fftn(ifftshift(cimdat))) ;

