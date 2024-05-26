function ckdat = i2ksbs(cimdat)
% I2KSBS I2K Slice by slice
%
% ckdat = i2ksbs(cimdat)
%
% David Atkinson
% %W% , created %G%
% See also K2I

ckdat = fftshift(fft(fft(ifftshift(cimdat),[],2),[],1)) ;
