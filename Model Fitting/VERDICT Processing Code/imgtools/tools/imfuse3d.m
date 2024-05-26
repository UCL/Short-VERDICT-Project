function C = imfuse3d(A,B,varargin)
% IMFUSE3D  Assembles an RGB image from two volumes
%
% C = imfuse3d(A,B)
%   A and B are 3D volumes of the same size
%   C is an RGB image composed of a red A image and green B 
% 
% Example
%  C = imfuse3d(A,B)
%  eshow(C,'vdim',[2 2 5], 'isrgb', 1)
%
% David Atkinson D.Atkinson@ucl.ac.uk  
% See also IMFUSE
%

[nyA, nxA, nzA] = size(A) ;
[nyB, nxB, nzB] = size(B) ;

if nzA ~= nzB
    error(['Inputs must have same number of slices'])
end

if nxA ~= nxB  ||  nyA ~= nyB
    error(['Inputs must have same number of rows and columns'])
end

C = zeros([nyA nxA nzA 3]) ;
 for islice = 1: nzA
     % C_this = imfuse(A(:,:,islice),B(:,:,islice),'ColorChannels',[1 2 0]) ;
     C_this = imfuse(A(:,:,islice),B(:,:,islice),'falsecolor') ;
     C(:,:,islice,:) = reshape(C_this,[nyA nxA 1 3]) ;
 end
 