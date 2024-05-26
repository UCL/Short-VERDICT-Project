function explore_B0simul
%
%
% See also B0read

exc_lowB0 = false ; dupsample = false ; zero_lowB0 = true ; 

[B0MRSeriesWaterFatShift_scaled, B0wfs_hzpp_scaled, B0fat_shift_dir_lph, B0dout  ] = dwfs('EMR','B0 map') ;

[DWIMRSeriesWaterFatShift_scaled, DWIwfs_hzpp_scaled, DWIfat_shift_dir_lph, DWIdout  ] = dwfs('EMR','DWI') ;

disp('Select ref T2')
T2fn = dselect;
dT2 = dmfparse(T2fn) ;
[vT2,mT2] = d2mat(dT2,{'slice'},'op','fp') ;


% Read in B0 image volume
[vB0, mB0, mask] = B0read( B0dout ) ;

% Read in DWI
DWIinfo = datparse(DWIdout.Filename) ;
[vDWI, mDWI] = d2mat(DWIinfo,{'slice','bv'},'bv',0,'op','fp') ;
if dupsample == true
    warning('May also need scaling in hzpp')
    mDWIorig = mDWI ;
    mDWI.geom = geom_change_ps(mDWI.geom, [1 1]) ;
    [vDWI] = vresample(vDWI, mDWIorig, mDWI) ; 
end

% Find location in 'true' space where B0map grid points are
map_pH_lph = -B0fat_shift_dir_lph / B0wfs_hzpp_scaled ;

% Get the DICOM LPH coordinates 
[Xt, Yt, Zt] = dicom2intrinsic(mB0.geom, 'output', 'LPHcoords') ;

Xti = double( Xt - vB0 * map_pH_lph(1) ) ; % true irregular
Yti = double( Yt - vB0 * map_pH_lph(2) ) ;
Zti = double( Zt - vB0 * map_pH_lph(3) ) ;

% exclude from B0 map any dodgy points (where modulus is low)
if exc_lowB0 == true
    loc = mask > 0.5 ; % points to keep
else
    loc = mask > -1 ; % keep all
end
if zero_lowB0 == true
    loc0 = mask < 0.5 ;
    vB0(loc0) = 0 ;
end
Fti = scatteredInterpolant(Xti(loc), Yti(loc), Zti(loc), double(vB0(loc)) ,...
    'linear', 'none') ;

B0nom = Fti(Xt, Yt, Zt) ;

% Now want B0map at nominal points of the DWI.
% Get the DWI LPH coordinates
[XDt, YDt, ZDt] = dicom2intrinsic(mDWI.geom, 'output', 'LPHcoords') ;

% B0map warped to space of undistorted and interpolated at nominal DWI coords
B0mapt = Fti(XDt, YDt, ZDt) ; 
B0mapt(isnan(B0mapt)) = 0 ; 

dwi_pH_lph = -DWIfat_shift_dir_lph / DWIwfs_hzpp_scaled ;

% Calculate displacement field for true to DWI. 
D = zeros([size(vDWI,1) size(vDWI,2) size(vDWI,3) 3]) ;
D(:,:,:,1) = B0mapt * dwi_pH_lph(1) ; % D maps from regularly spaced B0map to distorted DWI
D(:,:,:,2) = B0mapt * dwi_pH_lph(2) ;
D(:,:,:,3) = B0mapt * dwi_pH_lph(3) ;

vDWI_unwarp = imwarp(vDWI, D) ;

% 
figure
sl = 8
imshow(vDWI(:,:,sl)), hold on
quiver(D(:,:,sl,1), D(:,:,sl,2),0)

[vT2r] = vresample(vT2, mT2, mDWI) ; 
eshow(vT2r)

eshow(mask)
eshow(B0mapt,'geom',mDWI.geom)
eshow(vDWI_unwarp, 'geom', mDWI.geom)
eshow(vDWI, 'geom', mDWI.geom)
eshow(vB0, 'geom', mB0.geom)
eshow(vDWI - vDWI_unwarp, 'geom', mDWI.geom)





