function colwheel
% COLWHEEL Generate a colour wheel. Projection of radius to sphere
% elevation angle is linear here.
%
% colwheel
%
% See the paper by Pajevic and Pierpaoli  MRM 42 (3) p526. (1999)
%
% This routine generates vectors in the Northern hemisphere
% 
% David Atkinson, University College London, D.Atkinson@ucl.ac.uk.
% See also vec2col

% displayed image is from square pixels, compute their coords and
% convert to polar
%
% compute cartesian 2D coords
% compute polar 2D
% compute spherical polar
% compute cartesian on sphere, radius 1.

nhalf = 64 ;
rad = 1 ;
coords = 1*[-nhalf:(nhalf-1)]./(nhalf) ; %[ -1 1]

[X2D,Y2D] = meshgrid(coords,-coords) ;


% my theta's are in the xy plane
% paper's thetav is the pi/2-angle from the xy plane to z

[theta2D, rho2D] = cart2pol(X2D,Y2D) ;  % rho2d [ 0 1]
                                        % theta2D [-pi,pi]
					
					% rho2D = 0 corresponds to
                                        % phi3D = pi/2
					%
                                        % phi3D [pi/2 0]

outofcircle = find(abs(rho2D)>rad) ;

theta3D = theta2D ;              % in plane angle unchanged by
                                 % projection
				 
rho3D = ones(size(theta3D)) ;    % surface of unit radius colour sphere

% below is the projection from sphere to 2D. Here using a linear
% relation between 2D radius and angle phi3D  - (pi/2-thetav) in paper

phi3D = -pi/2*rho2D + pi/2 ;
			      
[X3D, Y3D, Z3D] = sph2cart(theta3D, phi3D, rho3D) ;

vec = zeros(prod(size(X3D)),3) ;

vec(:,1) = X3D(:) ;
vec(:,2) = Y3D(:) ;
vec(:,3) = Z3D(:) ;

rgb = vec2col(vec,ones(length(coords).^2 , 1),0,1) ;

rgb(outofcircle,:) = 0 ;

rgb = reshape(rgb,[length(coords) length(coords) 3]) ;

figure('Name','colwheel')
image(rgb)
axis square
axis off


