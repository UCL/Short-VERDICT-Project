function cimdat = k2i(ckdat)
%K2I      Returns the complex image data, given complex k-space     
%         cimdat = k2i(ckdat) 
%        
%         Handles 2D or 3D inputs
%
%         Currently uses matlab IFFT, may be modified in future to 
%         go either way to maintain compatibility with UCL and UMDS
%         code
%
%         David Atkinson , UMDS  
%
% @(#)k2i.m	1.4 , created 08/10/98
%
%  See also I2K I2H H2K H2I K2H 
%

cimdat = fftshift(ifftn(ifftshift(ckdat))) ;

