function [imRGB, imAlphaRGB, imCmapRGB, imScalarRemap, alpha, opts] = blendImScalar(imBase, imScalar, scaleBase, opts)
% BLENDIMSCALAR 
%
% David Atkinson
%
% Requires Kovesi's colorcet and normailse functions
%
% See also lwi_ratio

arguments
    imBase
    imScalar  % scalar map (single or double) with values in [0 1]
    scaleBase
    opts.scalarAlphaRange = [0.8 1] ;
    opts.alphaOutputRange = [0 1] ;
    opts.alphaRemapGamma = 0.6
    opts.scalarCRange     = [0.8 1]
    opts.scalarRemapGamma = 0.6
    opts.sc_cmap = colorcet('L18','reverse', false) ;
end

catdim = ndims(imBase)+1 ;

if min(imBase(:) < 0 ) || max(imBase(:)>1)
    warning('imBase has values outsde 0 to 1')
end

if min(imScalar(:)<0) || max(imScalar(:)>1)
    warning('imScalar has values outside [0 1]')
end

% alpha 1 denotes full imScalar
alpha = imadjust(imScalar,opts.scalarAlphaRange, opts.alphaOutputRange, opts.alphaRemapGamma) ;

% scalars to colour
sc_cmap = opts.sc_cmap ;
ncmap = size(sc_cmap,1) ;

% remap scalars into range 
imScalarRemap = imadjust(imScalar,[opts.scalarCRange], [0 1], opts.scalarRemapGamma) ;

imScalarInd = uint16(round(imScalarRemap * ncmap + 0.5)) ;
imScalarInd(imScalarInd<1) = 1 ;
imScalarInd(imScalarInd>ncmap) = ncmap  ;

imScalarRGB = ind2rgb( imScalarInd, sc_cmap) ;

% convert imBase to RGB

imBaseRGB = cat(catdim, scaleBase(1)*imBase, scaleBase(2)*imBase, scaleBase(3)*imBase) ;

imRGB = (1-alpha).*imBaseRGB + (alpha).*imScalarRGB ;

imFlatRGB = ones([size(imBase,[1 2]) 3]) ;
imChequerRGB = imFlatRGB ;
imChequerRGB(1:4:end, 1:4:end,:) = 0 ;
imChequerRGB(2:4:end, 1:4:end,:) = 0 ;
imChequerRGB(1:4:end, 2:4:end,:) = 0 ;
imChequerRGB(2:4:end, 2:4:end,:) = 0 ;

imAlphaRGB = (1-alpha).*imChequerRGB + alpha .* imFlatRGB ;

% sineramp image from Kovesi
wavelen = 4 ;
amp = 12.5 ;
p = 2 ;

cols = size(imBase,2) ;
rows = size(imBase,1) ;

cycles = round(cols/wavelen);
cols = cycles*wavelen;
    
% Sine wave
x = 0:cols-1;
fx = amp*sin( 1/wavelen * 2*pi*x);
    
% Vertical modulating function
A = ([(rows-1):-1:0]/(rows-1)).^p;
im = A'*fx;
    
% Add ramp
ramp = meshgrid(0:(cols-1), 1:rows)/(cols-1);
im = im + ramp*(255 - 2*amp);
    
% Now normalise each row so that it spans the full data range from 0 to 255.
% This ensures that, at the lower edge of the image, the full colour map is
% displayed.  It also helps with the evaluation of cyclic colour maps though
% a small cyclic discontinuity will remain at the top of the test image.
for r = 1:rows
    im(r,:) = normalise(im(r,:));
end
% im = im * 255;

imRemap = imadjust(im, opts.scalarCRange, [0 1], opts.scalarRemapGamma) ;

imCmapRGB = ind2rgb(uint16(round(255*imRemap)), sc_cmap) ;

end