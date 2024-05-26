function create_test_dicom(imtype, val)
% CREATE_TEST_DICOM
% Creates Dicom file that can be used for testing geomtry reading and manipulations.
%
% create_test_dicom(imtype)
%
% imtype can be:
%   'mri_12'  slice 12 from the MATLAB inbuilt mri image
%   'zoneplate_128'  a 128x128 image using imzoneplate
%   'zoneplate_plus' a 128x128 zoneplate image with added crosshair and
%   lettering.
%
% D.Atkinson@ucl.ac.uk
%
% See also burn_text writeDicom
%

if nargin == 0
    imtype = 'mri_12' ;
end

switch imtype
    case 'mri_12'
        dat = load('mri') ;
        img = abs(single(dat.D(:,:,1,12))) ;
        
        % make a geom structure
        geom.IOP = [1 0 0 0 1 0] ; % axial image
        geom.IPP = [0 0 0] ;
        geom.Height = size(img,1);
        geom.Width = size(img,2) ;
        geom.PixelSpacing_HW = [ 2 2] ;
        geom.SliceThickness = 5 ;
        geom.MRAcquisitionType = '2D' ;
        fnstem = imtype ;
    
    case 'zoneplate_128'
        % uses imzoneplate from MathWorks file exchange
        img = 2000 * imzoneplate(128) ;
        geom.IOP = [1 0 0 0 1 0] ; % axial image
        geom.IPP = [0 0 0] ;
        geom.Height = size(img,1);
        geom.Width = size(img,2) ;
        geom.PixelSpacing_HW = [ 2 2] ;
        geom.SliceThickness = 5 ;
        geom.MRAcquisitionType = '2D' ;
        fnstem = imtype ;
        
    case 'zoneplate_plus'
        % zoneplate, with added lines and lettering
        img = 2000 * imzoneplate(128) ;
        
        % zero central region
        gridDC = DCgp(img) ;
        a = 3 ;
        img(gridDC(1)-a:gridDC(1)+a, gridDC(2)-a:gridDC(2)+a) = 0 ;
        
        maxim = max(img(:));
        
        % Set cross
        img(1:size(img,1),gridDC(2)) = maxim ;
        img(gridDC(1), 1:size(img,2)) = maxim ;
        
        % Add lettering
        % uses text2im which returns a 20x18 image
        img = burn_text(img, 'L', 'east') ;
        img = burn_text(img, 'R', 'west')  ;
        img = burn_text(img, 'A', 'north')  ;
        img = burn_text(img, 'P', 'south') ;
        
        geom.IOP = [1 0 0 0 1 0] ; % axial image
        geom.IPP = [0 0 0] ;
        geom.Height = size(img,1);
        geom.Width = size(img,2) ;
        geom.PixelSpacing_HW = [ 2 2] ;
        geom.SliceThickness = 5 ;
        geom.MRAcquisitionType = '2D' ;
        
        fnstem = 'zoneplus' ;
        
    otherwise
        error(['Unsupported input image type: ',imtype])
end

args = {'folder_name', '/Users/davidatkinson/matlab/imgtools/unittest',...
    'SeriesDescription', 'test_dicom', 'fnstem', fnstem,  ...
    'ImageComments', 'Created by create_test_dicom', ...
    'MRAcquisitionType', geom.MRAcquisitionType, ...
    'geom', geom} ;

writeDicom(img, 'positive', args{:}) ;


