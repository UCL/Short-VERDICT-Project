function explore_roi
% EXPLORE_ROI ROI drawing on DICOM images and copying.

fnT2 = dselect('message','Select T2') ;
fnD = dselect('message','Select DWI') ;

dT2 = datparse(fnT2) ;
dD = datparse(fnD) ;

[vT2,mT2] = d2mat(dT2,{'slice'},'op','fp') ;
[vD,mD] = d2mat(dD,{'slice','bv'},'bv',0,'op','fp');

figure('Name','T2')
slT2 = ceil(size(vT2,3)/2) ;

imshow(vT2(:,:,slT2),[])
roi = drawpolygon ;

points = roi.Position ; % stacked intrinsic x,y coordinates

% Convert to LPH

points_lph = dicom2intrinsic(mT2.geom(slT2),'output', 'stackedLPH', 'coords', points) ;

slD = ceil(size(vD,3)/2) ;


pointsD = dicom2intrinsic(mD.geom(slD),'output', 'stackedIntrinsic', 'coords', points_lph) 
figure('Name','D')

him = imshow(vD(:,:,slD),[]) ;
roiD = images.roi.Polygon(him.Parent,'Position',pointsD(:,[1 2])) ;
zpD = abs(pointsD(:,3)) ;
if max(zpD) > 1
    roiD.StripeColor= [1 0 0] ;
elseif max(zpD) > 0.5
    roiD.StripeColor= roiD.Color/2 ; 
end





