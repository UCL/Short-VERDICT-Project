function plane = plane_process(plane)
% PLANE_PROCESS Process a plane structure. Checks or sets plane vectors
%
% plane = plane_process(plane)
%
% Returns unit vector normal to plane and two unit vectors in the plane.
% Accepts either plane normal, two vectors in the plane, or three points
% in the plane (not co-linear).
%
% normal = v1u x v2u
%
% If both plane.v1u and plane.v2u exist and are each  length 3, 
%   they are set to unit vectors and plane.normal is set to (v1u x v2u)
%
% If plane.normal exists and is length 3,
%   it is normalised to a unit vector if necessary
%   the x,y or z axis direction least colinear with
%   the normal is used to set v1u and v2u.
%
% If plane.points exists and has three coordinates,
%   these are used to set v1u and v2u and function is called recursively .
%
%
% David Atkinson, MVL, Engineering Science, Oxford University
%   and Radiological Sciences, Guy's Hospital, KCL, London.
%
% @(#)plane_process.m	1.2 , created 10/26/00
% See also TOOLS

dpt = 0.9  ; % dot product threshold. (Below considered co-linear)
uvt = 1e-11 ; % vector norm should be within uvt of 1

if ~isstruct(plane) ; error([' Input must be a plane structure.']); end

if isfield(plane,'v1u') & isfield(plane,'v2u')
  % check they have the same  length
  if length(plane.v1u) == length(plane.v2u) & length(plane.v1u) == 3
    % check normalised
  
    nv1 = norm(plane.v1u) ;
    nv2 = norm(plane.v2u) ; 
 
    plane.v1u = plane.v1u ./ nv1 ; 
    plane.v2u = plane.v2u ./ nv2 ; 
  
    % check not co-linear
    if abs(sum(plane.v1u.* plane.v2u)) > dpt
      warning([ 'Plane vectors have a dot product greater than ', ...
           num2str(dpt),'  co-linear!' ])
    end

    plane.normal = cross(plane.v1u, plane.v2u) ;
    plane.normal = plane.normal ./norm(plane.normal) ;
  
    return
  end
end

if isfield(plane,'normal')
  if length(plane.normal) == 3
    nnorm = norm(plane.normal) ;
    
    plane.normal = plane.normal ./ nnorm ;
    
    % choose the x,y or z axis least co-linear with norm, from 
    % that choose v1u to be perpendicular to these two 
    % and v2u perp to normal and v1u
    
    orth(1).v = [0 1 0] ;
    orth(2).v = [1 0 0] ;
    orth(3).v = [0 0 1] ;
    
    for iax = 1:3
      dots(iax) = abs(sum(orth(iax).v .* plane.normal)) ;
    end
    
    [val, indmin] = min(dots) ; 
    indmin = indmin(1) ;        % in case two were equal 
  
    plane.v1u = cross(plane.normal, orth(indmin).v) ;
    plane.v1u = plane.v1u ./ norm(plane.v1u) ;
    
    plane.v2u = cross(plane.normal, plane.v1u) ;
    
    return
  end
end

if isfield(plane, 'points')
  if size(plane.points,1) == 3 & size(plane.points,2) ==3 
    a = plane.points(1,:) ;
    b = plane.points(2,:) ;
    c = plane.points(3,:) ;
    
    ab = b - a ;
    ac = c - a ;

    plane.v1u = ab ;
    plane.v2u = ac ;
    
    plane = plane_process(plane) ;
    return
  end
end


warning([ 'Unable to set plane variables.'])


