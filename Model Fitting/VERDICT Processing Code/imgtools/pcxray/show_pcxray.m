function show_pcxray(sl, nslices)
% SHOW_PCXRAY Show Phase Contrast Xray slices
%
% show_pcxray(slice_number, nslices)

folder = '/Users/davidatkinson/OneDrive - University College London/pcxray/recon' ;

fstem = 'slice_' ;
ext = '.tif' ;

imcent = double(imread(fullfile(folder, [fstem, num2str(sl, '%04d'), ext]))) ;
[ny nx] = size(imcent) ;
    
slcs = [1:nslices] - floor((nslices-1)/2) + sl ;

vol = zeros([ny nx nslices]) ;

isl = 0 ;
for jsln = slcs
    isl = isl + 1 ;
    vol(:,:,isl) = double(imread(fullfile(folder, [fstem, num2str(jsln, '%04d'), ext]))) ;
end
imav = sum(vol,3) / length(slcs) ;

eshow(imav,'Name',['Sum of ',num2str(nslices),' slices about slice: ',num2str(sl)])

eshow(vol,'Name', ['Slices about slice: ',num2str(sl)])







