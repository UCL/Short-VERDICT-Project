function [R2D, D] = geom2sro(geom)
% GEOM2SRO Create 2D spatial referencing object from DICOM style geom
% 
% [R2d, D] = geom2sro(geom)
%
% Spatial referencing object will be a plane in the same plane as geom and
% the "world" axes set to match the pixel spacing and an in-plane origin at
% the point in the plane closest to the geom origin. 
% D is the distance from the geom origin to the closest point in the plane.
%
% The issue here is that the MATLAB spatial referencing objects need to
% have the same axes directions as the image, hence we can't just use the
% DICOM stuff directly. The "world" axes in the SRO are not the DICOM ones
% - they are a projection.
%
% D.Atkinson@ucl.ac.uk
% See also DGEOMEXTRACT
%

% See http://mathworld.wolfram.com/Plane.html

IPP = geom.IPP; IOP = geom.IOP;

n = cross(IOP(1:3), IOP(4:6) ) ; % Plane normal

d = -dot(n,IPP) ; % n = (a,b,c), ax+ by + cy + d = 0 

D = d/norm(n,2) ; % Distance from origin to plane. (uses 2 norm)

x0 = -D.*n ; % Point in plane, closest to origin

%Range in world coords
topl = IPP + -0.5*(geom.PixelSpacing_HW(2)*IOP(1:3) + ...
                    geom.PixelSpacing_HW(1)*IOP(4:6));
                
topr = IPP + (geom.Width-0.5)*(geom.PixelSpacing_HW(2)*IOP(1:3)) + ...
              -0.5*geom.PixelSpacing_HW(1)*IOP(4:6); 
          
botl = IPP -0.5*geom.PixelSpacing_HW(2)*IOP(1:3) + ...
    (geom.Height-0.5)*geom.PixelSpacing_HW(1)*IOP(4:6) ;

% used only for drawing:
botr = IPP + (geom.Width-0.5)*(geom.PixelSpacing_HW(2)*IOP(1:3)) + ...
    (geom.Height-0.5)*geom.PixelSpacing_HW(1)*IOP(4:6) ;

% Shift in plane origin to be at point in plane closest to true origin.
topls = topl - x0;
toprs = topr - x0 ;
botls = botl - x0 ;

% project into plane 
WPX = [dot(topls,IOP(1:3)) dot(toprs,IOP(1:3)) ] ;
WPY = [dot(topls,IOP(4:6)) dot(botls, IOP(4:6)) ] ;

R2D = imref2d([geom.Height geom.Width],WPX, WPY) ;

return

% TEST SECTION
figure('Name','geom2sro_test')
plot3(IPP(1),IPP(2),IPP(3),'ro')
text(IPP(1),IPP(2),IPP(3),'IPP')
hold on
plot3(0,0,0,'k+')
text(0,0,0,'0')
plot3([0 x0(1)],[0 x0(2)],[0 x0(3)])
plot3(x0(1),x0(2),x0(3),'kx')
text(x0(1),x0(2),x0(3),'x0')
plot3([topl(1) topr(1) botr(1) botl(1) topl(1)], ...
      [topl(2) topr(2) botr(2) botl(2) topl(2)], ...
      [topl(3) topr(3) botr(3) botl(3) topl(3)] )
grid on
axis square
axis equal
xlabel('L'),ylabel('P'),zlabel('H')

figure('Name','2D plane')
plot3([ WPX(1) WPX(2) WPX(2) WPX(1) WPX(1)], ...
    [WPY(1) WPY(1) WPY(2) WPY(2) WPY(1)], ...
    [D D D D D])
grid on
axis square
axis equal
xlabel('X'),ylabel('Y'),zlabel('Z')





