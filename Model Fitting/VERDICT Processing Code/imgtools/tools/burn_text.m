function im2write = burn_text(im2D, text, location) 
% BURN_TEXT
% Burns text into an image.
% 
%  im2write = burn_text(im2D, text, location) 
%
% location can be
%  'northeast'
%  'west', 'east', 'north', 'south'
%
% Adapted from code in writeDicom, uses text2im from MATLAB Central
% text2im returns a 20 (h) x 18 (w) image
%
% See also writeDicom text2im

im2write = im2D ;
if isempty(text)
    return
end
[ny, nx, nz] = size(im2D) ;

imtext = text2im(text) ;
[nyt, nxt] = size(imtext) ;

ys = ny/nyt; xs = nx/nxt ;

scale = min([1 ys xs]) ;
imtexts = imresize(imtext,scale) ; 
imtexts = mat2gray(imtexts) ; % removes slight under or overshoot from interp
    
rng = double(max(im2D(:)) - min(im2D(:))) ;
if rng == 0
    rng = 1;
end
imtexts = imtexts*rng + double(min(im2D(:))) ; % double

imtexts = cast(imtexts, 'like', im2D) ;

[ht, wt] = size(imtexts) ;

switch location
    case 'northeast'
        llr = 1;
        llc = nx - wt -1 ;
    case 'north'
        llr = 1 ;
        llc = max([1 floor(nx/2 - wt/2)]) ;
    case 'east'
        llr = floor(ny/2 - ht/2) ;
        llc = nx - wt -1 ;
    case 'west'
        llr = floor(ny/2 - ht/2) ;
        llc = 1 ;
    case 'south'
        llr = ny -ht+1; 
        llc = max([1 floor(nx/2 - wt/2)]) ;
    otherwise
        error(['not implemented location: ',location])
end

im2write(llr:llr+ht-1, llc:llc+wt-1) = imtexts ;

end