function test_imwarp
% TEST_IMWARP
%

disp(['Select a DICOM for testing'])
fn = pref_uigetfile('test_dresamp','fn') ;
dinfo = dsfread(fn) ; 
[data, gorig] = d2mat(dinfo, {'slice'} ) ;

geom = gorig.geom ;

iptsetpref('ImshowBorder','loose')
figure
imshow(mat2gray(data),geom.R2D)

ang = 30 ;
tform = affine2d([cosd(ang) -sind(ang) 0; sind(ang) cosd(ang) 0; 0 0 1]);
        
[B, RB] = imwarp(data, geom.R2D, tform) ;

figure
imshow(mat2gray(B),RB)