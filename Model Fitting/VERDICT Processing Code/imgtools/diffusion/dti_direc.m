%% DTI Directions
% Used in sequence design for optimal directions
%
% N=6 Icosahdral
% Wikipedia
%
% phi = (1+sqrt(5))/2 ;
%
% (0, 1, phi)
% (0, 1, -phi)
% (1, phi, 0)
% (1, -phi, 0)
% (phi, 0, 1)
% (phi, 0, -1)
%

phi = (1+sqrt(5))/2 ;

grad6 = [ ...
    phi 0 1
    1 phi 0
    1 -phi 0
    0 1 phi 
    0 1 -phi 
    phi 0 -1 ] ;

row_norm = norm(grad6(1,:)) ;

% normalise
grad6 = grad6 / row_norm ;

%rotate so that first is [0 0 1]
r_ang = atan(-phi) ;

M = makehgtform('yrotate',r_ang)  ;
rgrad6 = M * transpose(cat(2,grad6,ones([6 1]))) ;
grad6 = transpose(rgrad6(1:3,:)) ;





