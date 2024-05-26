function cimdat = k2isbs(ckdat)
% K2ISBS K2I Slice by slice
%
% cimdat = k2isbs(ckdat)
%
% David Atkinson
% %W% , created %G%
% See also K2I

cimdat = fftshift(ifft(ifft(ifftshift(ckdat),[],2),[],1)) ;
