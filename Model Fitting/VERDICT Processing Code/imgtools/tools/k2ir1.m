function cimdat = k2ir1(ckdat, dim)
%K2IR1    Performs 1D FFT transform in the RECON k to i direction. 
%         NOTE the FFT direction differs from convention used previously!!    
%         ckdat = k2ir1(cimdat, dim) 
%        
%         Handles multi-dimensional inputs
%
%         Uses matlab IFFT.
%         K -> I FFT:   1 \sum exp(-)
%         I -> K IFFT:  1/N \sum exp(+)
%
%         David Atkinson
%
%  See also I2KR1 I2K I2H H2K H2I K2H 
%

cimdat = fftshift(fft(ifftshift(ckdat, dim), [], dim), dim) ;

