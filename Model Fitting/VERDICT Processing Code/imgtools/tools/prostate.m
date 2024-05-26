function [img, E] = prostate(n)
% PROSTATE phantom
%
% [img, E] = prostate(n)
%
% D.Atkinson@ucl.ac.uk
%
%
% See also phantom
%

% Column
% Parameter
% Column 1  A	
% Additive intensity value of the ellipse
% Column 2  a	
% Length of the horizontal semiaxis of the ellipse
% Column 3 b	
% Length of the vertical semiaxis of the ellipse
% Column 4 x0	
% x-coordinate of the center of the ellipse
% Column 5 y0	
% y-coordinate of the center of the ellipse
% Column 6 phi	
% Angle (in degrees) between the horizontal semiaxis of the ellipse and the x-axis of the image

%               A    a     b    x0    y0     phi
bladder =    [ 0.5  1/5   1/6   0      0.2    0 ];
skin    =    [ 0.5  0.9   0.7   0      0      0 ] ;
deskin  =    [ -0.3 0.85  0.65  0      0      0 ] ;
prostate=    [ 0.25  1/7   1/8   0     -0.15   0 ] ; 
deprostate = [ -0.1 1/7  1/10   0     -0.13   0 ] ;
ureter  =    [ 0.3  0.02  0.02  0.02  -0.2   0 ] ;
rectum  =    [ -0.15 1/9   1/9   -.05   -0.4   20 ] ;
cyst    =    [ 0.3  0.02  0.02  -0.07   -0.1    0 ] ;

E = cat(1, bladder, skin, deskin, prostate, deprostate, ureter, rectum, cyst) ;

img = phantom(E,n) ;
