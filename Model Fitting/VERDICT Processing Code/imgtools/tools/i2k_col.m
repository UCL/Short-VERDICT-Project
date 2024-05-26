function ckdat = i2k_col(cimdat)
%I2K_COL  Returns the complex k-space data, given complex image domain column vectors.
%         ckdat = i2k_col(cimdat) 
%
%         Currently uses matlab FFT, may be modified in future to 
%         go either way to maintain compatibility with UCL and UMDS
%         code
%
%         I -> K FFT:   1 \sum exp(-)
%         K -> I IFFT:  1/N \sum exp(+)
%         David Atkinson , UCL 
%


ckdat = fftshift(fft(ifftshift(cimdat,1)),1) ;

