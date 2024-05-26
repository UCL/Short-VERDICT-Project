function test_dresamp(test_type)
% TEST_DRESAMP
%
%  test_dresamp(test_type)
%
% Testing of dresamp or dreslice, 
% 1) Order of rotations and scale changes in dresamp
% 2) Non-isotropic kernel and scalings.
%
% Create a DICOM with known properties.
% 1) Make a copy and scale and rotate using dinplanet
%    Call dresamp to get the new into the space of the original.
%    Check with order of rotations and scales set in dresamp.
%
% 2) Check the kernel scalings
%
%
% D.Atkinson@ucl.ac.uk
%
% See also create_test_dicom
%

disp(['Select a DICOM for testing'])
fn = pref_uigetfile('test_dresamp','fn') ;
dinfo = dsfread(fn) ; 
[data_orig, gorig] = d2mat(dinfo, {'slice'} ) ;

switch test_type
    case 'scale'
        % scale
        [ data_test, geom_test ] = dinplanet( data_orig, gorig.geom, 'imresize', [128 32] ) ;
        
    case 'rot+1'
        [ data_test, geom_test ] = dinplanet( data_orig, gorig.geom, 'rot', 1 ) ;
        
    case 'rot-1'
        [ data_test, geom_test ] = dinplanet( data_orig, gorig.geom, 'rot', -1 ) ;
        
    case 'scale_rot'
        % scale
        [ data_rs, geomt_rs ] = dinplanet( data_orig, gorig.geom, 'imresize', [128 64] ) ;
        % rotate
        [ data_rs_rot, geom_rs_rot ] = dinplanet( data_rs, geomt_rs, 'rot', 1 ) ;
        data_test = data_rs_rot ; geom_test = geom_rs_rot ;
        
    case 'circshift'
        % shift
        [ data_test, geom_test] = dinplanet( data_orig, gorig.geom, 'circshift', [12 5] ) ;
        
    case 'rot30'
        % For angles less than 45 and for axial slice only
        if norm(gorig.geom.IOP(:)-[1 0 0 0 1 0]') > 0.01
            error(['rot30 is only for axial inputs'])
        end
        
        ang = 30 ;
        tform = affine2d([cosd(ang) -sind(ang) 0; sind(ang) cosd(ang) 0; 0 0 1]);
        
        [B, RB] = imwarp(data_orig, gorig.geom.R2D, tform) ;
        % with IPP being at 0,0,0 in the test data, this effectively
        % rotates about IPP. The output matrix is larger and has its limits
        % speicified in XWorldLimits. WRT the original image, these are in
        % a coordinate system that is rotated. 
        
        % assuming [1 0 0 0 1 0] input
        geom_test = gorig.geom ;
        geom_test.IOP = [cosd(ang) sind(ang) 0 -sind(ang) cosd(ang) 0] ;
        
        % Concept is that the object stays still but the camera rotates.
        % After imwarp, the image up and right directions will be:
        up = [sind(ang) -cosd(ang) 0] ;
        right = [cosd(ang) sind(ang) 0] ;
        
        % The new top left corner is 
        tlc = [0 0 0] +  right*RB.XWorldLimits(1) - up*RB.YWorldLimits(1) ;
        
        %geom_test.IPP = tlc + 0.5*(geom_test.IOP(1:3)*RB.PixelExtentInWorldX + ...
        %    geom_test.IOP(4:6)*RB.PixelExtentInWorldY) ;
        
        
        geom_test.IPP = tlc + 0.5*(right*gorig.geom.R2D.PixelExtentInWorldX + ...
            -up*gorig.geom.R2D.PixelExtentInWorldY) ;
        
        geom_test.R2D = RB ; % this might not be correct wrt image
        geom_test.Height = size(B,1);
        geom_test.Width = size(B,2) ;
        
        data_test = B ;
        
        
        % now add a scaling!
        [ data_test, geom_test ] = dinplanet( data_test, geom_test, 'imresize', [512 200] ) ;
        
    otherwise
        error(['Unknown test: ',test_type])
end

%resample
[vout, gout] = dresamp(data_test, geom_test, gorig.geom) ;

figure('Name',['(test_dresamp) ',test_type])
imshowpair(data_orig, vout)





