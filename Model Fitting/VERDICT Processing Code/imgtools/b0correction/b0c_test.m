function [Iwarped, vtowarp, vstrucatwarp] = b0c_test(varargin)
% B0C_TEST Use B0 map to undistort an image
%  Requires Philips multiframe DICOM and XX DICOM files.
% 
%  Iwarped = b0c_test ; % calls ui for inputs
%
% Parameter / value pairs  [default]
%  'nlwm'  [10]  fitgeotrans parameter
%  'PPM_WFS'  [3.4]  Water Fat Shift (3.4 used by Philips)
%  'struct_comp' [true]  Loads a structural image for comparison (for
%     checking)
%  'Dtype' ['EMR'] or 'Single'  for Enhanced MR or Single Frame
%
% To Do
% Rethink of transformations and how they are applied.
% May be possible to compute transform once and apply forward and back as
% necessary and in appropriate directions. Seems like the unwarping should
% invovle an inverse transformation, yet one is not used here.
% Some extrapolation/masking of B0 map would be useful at edges?
% Test with range of WFS directions, including shifted B0
%
%
% Example
%  
%  David Atkinson   D.Atkinson@ucl.ac.uk
%
% See also B0shift dwfs  contour_disp
 

% Water Fat Shift in PPM is 3.4 in Philips code.
PPM_WFS = 3.4 ;

% Parameter for fitgeom (6 is minimum for 2nd degree polynomial)
nlwm = 10 ;

struct_comp = true ;
Dtype = 'EMR' ;

for ip = 1:2:length(varargin)
    switch varargin{ip}
        case 'nlwm'
            nlwm = varargin{ip+1} ;
        case 'PPM_WFS'
            PPM_WFS = varargin{ip+1} ;
        case 'struct_comp'
            struct_comp = varargin{ip+1} ;
        case 'Dtype'
            Dtype = varargin{ip+1} ;
        otherwise
            error(['Unknown parameter: ',varargin{ip}])
    end
end

% Select B0 map then image to warp.
[B0_wfs, B0_wfs_hzpp, B0_wfs_dir, dinfB0] = dwfs(Dtype, 'B0') ;

[towarp_wfs, towarp_wfs_hzpp, towarp_wfs_dir, dinftowarp] = dwfs(Dtype, 'towarp') ;

% Select structural image for comparison
if struct_comp
    [struc_wfs, struc_wfs_hzpp, struc_wfs_dir, dinfstruc] = dwfs(Dtype, 'Structural image') ;
    dstruc = datparse(dinfstruc.Filename) ;
    [vstruc,mstruc] = d2mat(dstruc,{'slice'},'op','fp') ;
end


% Read in B0map
disp(['Reading in B0 data'])
dB0 = datparse(dinfB0.Filename) ;

% Annoying changes in use of ImageType. For Philips Release 5.1.7, seems to be 20,
% whether EMR or Single? 

dB0uq = unique([dB0.itype]) ;
if ~isempty(intersect(dB0uq,20))
    itypeb0 = 20 ;
else
    itypeb0 = 9 ;
end

[B0regf0, mB0] = d2mat(dB0,{'slice','itype'},'itype',itypeb0,'op','dv') ;

% switch Dtype
%     case 'EMR'
%         [B0regf0, mB0] = d2mat(dB0,{'slice','itype'},'itype',9,'op','dv') ;
%     case 'Single'
%         [B0regf0, mB0] = d2mat(dB0,{'slice','itype'},'itype',20,'op','dv') ;
% end

B0regf0 = double(B0regf0) ;

%Read in image towarp
disp(['Reading in towarp data'])
dtowarp = datparse(dinftowarp.Filename) ;
[vtowarp, mtowarp] = d2mat(dtowarp, {'slice','bv'},'bv',0,'op','fp') ;

%reslice B0 to planes of towarp, keeping resolution
% [B0rs, mB0rs] = dreslice(B0regf0, mB0, mtowarp, 'PixelSpacing','input') ;

% Compute shifts for B0 map itself
[Xs, Ys, X, Y] = B0shift( B0regf0, B0regf0, dB0(1).ImageOrientationPatient, ...
    B0_wfs_hzpp, B0_wfs_dir ) ;

B0true = zeros(size(B0regf0)) ;
for islice = 1: size(B0regf0,3) 
    B0true(:,:,islice) = griddata(Xs(:,:,islice), Ys(:,:,islice), ...
        B0regf0(:,:,islice), X(:,:,islice), Y(:,:,islice)) ;
end
B0true(isnan(B0true)) = 0 ;
eshow(B0true)

[B0atwarp] = dreslice(B0true, mB0, mtowarp);

if struct_comp
    vstrucatwarp = dreslice(vstruc, mstruc, mtowarp) ;
else
    vstrucatwarp = [] ;
end


% Compute shifts
[Xs, Ys, X, Y] = B0shift( B0atwarp, vtowarp, dtowarp(1).ImageOrientationPatient, ...
    towarp_wfs_hzpp, towarp_wfs_dir ) ;


nslice = size(B0atwarp,3) ;
Iwarped = zeros(size(B0atwarp)) ;



% !!! Need to sort order of moving and fixed, directions above and spatial
% referencing (or use other tformarray)

% Apply shifts
hw = waitbar(0,['Warping ',num2str(nslice),' slices']) ;

for islice = 1: nslice  % nslice
    waitbar(islice/nslice, hw) ;
    Xthis = X(:,:,islice); Ythis = Y(:,:,islice); Ysthis=Ys(:,:,islice) ; Xsthis=Xs(:,:,islice) ;
    fixedPoints = cat(2,Xthis(:),Ythis(:)) ;
    movingPoints = cat(2,Xsthis(:),Ysthis(:)) ;

    tform = fitgeotrans(movingPoints,fixedPoints,'lwm',nlwm) ;
    % tform = fitgeotrans(movingPoints,fixedPoints,'pwl') ;
    
    R = imref2d(size(vtowarp(:,:,1))) ;
    Iwarped(:,:,islice) = imwarp(vtowarp(:,:,islice), tform, 'OutputView', R) ;
end
close(hw)

eshow(Iwarped)

