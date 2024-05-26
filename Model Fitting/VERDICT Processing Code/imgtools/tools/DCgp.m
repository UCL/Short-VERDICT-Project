function grid_DC = DCgp(x,dim)
% Return the DC grid point as a 3D coordinate.
% The DC grid point is based solely on the size of the input array x.
% The true k-space DC (location of max) can be found using dc3D
%
% grid_DC = DCgp(x)   grid_DC is length max(3,ndims(x), dim)
% grid_DC = DCgp(x,dim)  - return dimension dim only.
%
%
% David Atkinson, August, 1998.
% The Guy's, King's and St. Thomas' School of Medicine
% @(#)DCgp.m	1.2 , created 12/21/98
% 
% See also DC3D

if nargin == 1
  numDims = max([3 ndims(x)]) ; 
else
    numDims = max([3 ndims(x) dim]) ;
end

grid_DC = zeros(1,numDims) ;

for idim = 1:numDims
  nx = size(x,idim) ;
  if rem(nx,2) == 0
    grid_DC(idim) = nx/2 + 1 ;
  else
    grid_DC(idim) = (nx+1)/2 ;
  end
end

if nargin == 2
  grid_DC = grid_DC(dim) ;
end
