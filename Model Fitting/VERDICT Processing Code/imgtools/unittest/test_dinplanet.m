function test_dinplanet
% TEST_DINPLANET
%
%

dinfo = dparse ;

[data, g] = d2mat(dinfo, {'slice'} ) ;

[ datat, geomt ] = dinplanet( data, g.geom, 'imresize', [128 64] ) ;
disp(['After resizing to 128 64'])
geomt

[ datat, geomt ] = dinplanet( datat, geomt, 'imresize', [128 128] ) ;
disp(['After resizing to 128 128'])
geomt

imshowpair(data, datat)
