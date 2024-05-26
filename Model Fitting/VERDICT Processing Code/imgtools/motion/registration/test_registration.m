% test_registration
%
% Load DICOM slices
% Check on IOPs and Ds
% Pass to imregtfrom
% Look at output
% Try different scales and resampled data
% Test applying imwarp
%
% Use imshowpair as this uses spatial referencing objects
% See also imregtfrom imwarp imregister imshowpair

d1 = datparse(dselect('ismodal',true)) ;
d2 = datparse(dselect('ismodal',true)) ;

[v1,m1] = d2mat(d1,{'slice'},'op','fp') ;
[v2,m2] = d2mat(d2,{'slice'},'op','fp') ;

eshow(v1)
eshow(v2)

s1 = input('Slice from v1: ') ;
s2 = input('Slice from v2: ') ;

im1 = v1(:,:,s1) ;
im2 = v2(:,:,s2) ;

g1 = m1.geom(s1) ;
g2 = m2.geom(s2) ;

figure
imshowpair(im1,g1.R2D,im2,g2.R2D)

transformType = 'similarity';
modality = 'monomodal' ;
[optimizer,metric] = imregconfig(modality) ;

tform = imregtform(im2,g2.R2D,im1,g1.R2D,transformType,optimizer,metric) ;


[B,RB] = imwarp(im2,g2.R2D,tform,'cubic') ;
figure
imshowpair(im1,g1.R2D, B,RB)




