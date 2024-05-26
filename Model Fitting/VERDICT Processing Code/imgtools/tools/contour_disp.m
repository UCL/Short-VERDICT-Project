function contour_disp(vin, vcont, islice)
% CONTOUR_DISP Displays contours from one image on another
%  useful for checking positional shifts between two images
%
%  contour_disp(vin, vcont, islice)
%  contour_disp(vin, vcont)  chooses mid slice
%
% vin and vout must have the same size (possibly after dreslice), may be 
% volumes but only one slice shown
% 
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also dreslice, B0

if ~isequal(size(vin), size(vcont))
    error(['Input volumes must be same size.'])
end

nz = size(vin,3) ;

if nargin <3
    islice = ceil(nz/2) ;
end

im = mat2gray(vin(:,:,islice)) ;
imc = mat2gray(vcont(:,:,islice)) ;

figure('Name',['contour_disp: im: ',inputname(1), ', cont: ',inputname(2),', slice: ',num2str(islice)])
imshow(im)
hold on
imcontour(imc,'r')



