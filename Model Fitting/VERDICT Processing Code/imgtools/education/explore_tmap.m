function explore_tmap

dinfoV = dmfparse(dselect) ;
dinfoQ = dmfparse(dselect) ;

[vV, mV] = d2mat(dinfoV,{'slice'},'op','fp');
[vQ, mQ] = d2mat(dinfoQ,{'slice'},'op','fp');
% warning(['faking vV'])
% vV = repmat(imzoneplate(size(vV,1)),[1 1 length(mV.geom)]) ;
% mQ.geom(1).IPP = mQ.geom(1).IPP + [0.95 0.95 0]'/2 ;

mQ_orig = mQ ;
mQ.geom = geom_change_ps(mQ.geom, mV.geom(1).PixelSpacing_HW) ;

[drv, drm] = dreslice(vV,mV,mQ_orig,'PixelSpacing','input') ;

B = vresample(vV, mV, mQ) ;


eshow(vV, 'geom', mV.geom)
eshow(vQ, 'geom', mQ.geom)
eshow(B,  'geom', mQ.geom)
eshow(B-drv, 'Name','b-drv', 'geom', mQ.geom)
