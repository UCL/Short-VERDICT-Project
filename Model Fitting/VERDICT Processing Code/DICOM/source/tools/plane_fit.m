function [NU, P, d] = plane_fit(coords3D)
% Fit 3D coordinates to a plane
%
%  [NU, P, d] = plane_fit(coords3D) 
%      coords3D [N x 3]
%
%      NU is unit normal vector [3 1]
%      P  is a point in the plane (the centroid of the points)
%      d  is distance to origin (see below)
%
% Equation of plane is:
%  ax + by + cz + d = 0
%
%  where a = NU(1), b=NU(2), c = NU(3)
%  d = -( aP(1) + bP(2) + cP(3) ) 
%
% when the noraml is unit (as here), d is the distance of the plane from
% the origin
% 
% In Hessian Normal Format
% dot(NU.X) = - p   
%   where X is a coord in the plane
%   p is the distance of the plane from the origin (i.e. +/- d)
%   In Hessian normal form, if p > 0, the origin is in the half-space
%   pointed to by NU
%
%  The point-plane distance, D, from a point X0 is
% D = dot(NU,X0) + p
%
% If X0 is in the half-space pointed to by NU, then D is positive
%
%
% Also see 
%   https://mathworld.wolfram.com/HessianNormalForm.html
%   https://mathworld.wolfram.com/Coplanar.html
%   https://www.ltu.se/cms_fs/1.51590!/svd-fitting.pdf

% coords3D [N x 3]

P = mean(coords3D,1) ; % [1 3] centroid
mcoords3D = coords3D - P ; 

[U,S,V] = svd(mcoords3D,0);

% Should be rank 2 or less to be coplanar in 3D
% See Coplanar in Wolfram

% This is a test of rank (3rd SV less than 1000th of first SV)
if S(3,3)>S(1,1)/1000 
    warning('MATLAB:plane_fit:NonCoplanar','Points not coplanar')
end

N = V(:,end) ;

NU = N ./ norm(N) ; % N may be unit anyway by construction?

if nargout > 2
    d = -(dot(NU,P)) ;
end

end