function ckdat = i2kr1(cimdat, dim)
%I2KR1    Performs 1D IFFT transform in the RECON i to k direction. 
%         NOTE the FFT direction differs from convention used previously!!    
%         ckdat = i2kr1(cimdat, dim) 
%        
%         Handles multi-dimensional inputs
%
%         Uses matlab IFFT.
%         K -> I FFT:   1 \sum exp(-)
%         I -> K IFFT:  1/N \sum exp(+)
%
%         David Atkinson
%
%  See also I2K I2H H2K H2I K2H 
%

ckdat = fftshift(ifft(ifftshift(cimdat, dim), [], dim), dim) ;

