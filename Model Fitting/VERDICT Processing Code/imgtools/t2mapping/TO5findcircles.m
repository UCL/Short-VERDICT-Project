function [rois, roi_fluid] = TO5findcircles(img, pixs)
% TO5findcircles Find locations of gels and whole TO5 phantom
%
% [rois, roi_fluid] = TO5findcircles(img, P)
%
% img 2D image of the TO5 phantom, slots should be reasonably straight.
%
% P can be scalar (the pixel size in mm), a two element vector
%   with equal elements, a structure from d2mat, or a geom structure.
%
% The rois are found in no particular order. They are labelled here 
% with a letter to indicate which slot they represent.
%
% Slots:
%
%        A  B
%     C  D  E  F    and Z for the fluid at the centre
%     G  H  I  J
%        K  L
%
% To avoid ringing, the rois for the gels are 8mm radius.
%
% D.Atkinson@ucl.ac.uk
%
% See also TO5quant
%

% TO5 has a radius of about 96mm
% There are 12 gel spaces, each gel tube has a radius of about 11mm

img = mat2gray(img) ;

rTO5 = 96 ; % phantom radus in mm
rgel = 11 ; % gel radius in mm

rroi = 8 ;

% Location of gel centres wrt centre of phantom (assume aligned)
% [x y] in mm wrt centre
gcent = [ ...
    -21.985      -64.96 ;
     20.991      -65.26 ;
    -66.325      -21.133 ;
    -22.455      -21.338 ;
     21.327      -21.847 ;
     65.415      -21.823 ;
    -65.859       22.717 ;
    -22.267       22.633 ;
     22.073       21.878 ;
     65.248       22.219 ;
    -22.183       66.712 ;
     21.893       66.023 ] ;


% Allow input as either a single value, a two element array or a structure
% with a geom

if isstruct(pixs)
    if isfield(pixs,'geom')
        PS = pixs.geom.PixelSpacing_HW ;
    else
        PS = pixs.PixelSpacing_HW ; 
    end
else
    PS = pixs ;
end

if length(PS) == 2
    if abs(PS(2)-PS(1))/PS(1) > 0.001
        error('Pixels need to be square')
    end
end

pix = PS(1);

% Find the phantom
[centTO5, radTO5] = imfindcircles(img, round(rTO5./pix.*[0.9 1.1]), 'ObjectPolarity', 'bright') ;
if isempty(centTO5)
    disp(['No phantom found.'])
else
    disp(['Phantom located with radius ',num2str(radTO5*pix),' mm.'])
end

% Find the gels
[centgel, radgel] = imfindcircles(img,round(rgel/pix*[0.6 1.4]),'ObjectPolarity','bright', 'Sensitivity',0.9);
ngel = length(radgel) ;
disp([num2str(ngel),' gels found.'])

figure('Name','TO5findcircles')
imshow(img)
% viscircles(centTO5, radTO5)
% viscircles(centgel, radgel)

for igel = 1: ngel
    % Work out which slot this is (they are not found in order)
    thiscent = (centgel(igel,:) - centTO5)*pix ;
    centdist = gcent - repmat(thiscent,[size(gcent,1) 1]) ;
    dists = vecnorm(centdist,2,2) ;
    [mv, mloc] = min(dists) ;
    
    rois{igel} = images.roi.Circle(gca,'Center',centgel(igel,:), ...
        'Radius',rroi,'Label',char(64+mloc)) ;
end

roi_fluid = images.roi.Circle(gca,'Center',centTO5, ...
        'Radius',rroi,'Label','Z') ;
    
drawnow


    