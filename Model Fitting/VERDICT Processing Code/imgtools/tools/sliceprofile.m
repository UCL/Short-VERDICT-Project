function sliceprofile
% SLICEPROFILE Measure slice profile 
%
% sliceprofile
%
% Asks for DICOM file (volume of phantom with "plate" of fluid such as the
% Philips 200mm head coil phantom)
% Displays the volume to let user choose the relevant slice number.
% Displays that slice with imcontrast gui to allow re-windowing.
% User clicks start of profile and double click at end.
% Plotted profile accounts for wedge angle specified in code (currently set
% at 11.3 degrees) - should be checked.
% 
% Determine slice profile from wedge phantom
%
% D.Atkinson@ucl.ac.uk

wedge_angle = 12  % wedge angle in degrees
disp('Select DICOM file')
dinfo = datparse(dselect) ;

[vp, mp] = d2mat(dinfo,{'slice'},'op','fp') ;
XData = mp.geom.XData ;
YData = mp.geom.YData ;

disp('Choose slice number')
eshow(vp,'Name','choose slice number')

sl = input('Enter slice number: ') ;

figure('Name','Draw profile')
himshow = imshow(vp(:,:,sl),[],'XData',XData,'YData',YData) ;
imcontrast(himshow)
[cx,cy,c,xi,yi] = improfile ;

figure('Name','slice profile')
r=sqrt((cx-xi(1)).^2 +(cy-yi(1)).^2);

r=r*tan(wedge_angle/360*2*pi) ; % scale by wedge angle

plot(r,c)
grid on
xlabel('Distance/ mm')

