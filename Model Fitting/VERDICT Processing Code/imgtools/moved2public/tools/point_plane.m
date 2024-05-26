function d = point_plane(point, plane) 
% POINT_PLANE Point to plane distance
%
% d = point_plane(point, plane)
% 
% plane is a structure produced by set_plane
%
% see MathWorld plane, also, Foley and Van Dam A.3.4
%
% David.Atkinson@kcl.ac.uk
%
%

d = (dot(plane.abcd(1:3), point) + plane.d) / norm(plane.abcd(1:3))  ;


