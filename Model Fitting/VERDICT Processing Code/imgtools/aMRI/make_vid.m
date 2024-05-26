function make_vid
% MAKE_VID Make video of phantom to trial aMRI 
%

dT = 0.5 ; % frame time in s
nF = 100 ;

def = 'Modified Shepp-Logan' ; n= 128 ;

[P, E] = phantom(def,n);
elip = 4 ;

% Column 1 A	Additive intensity value of the ellipse
% Column 2 a Length of the horizontal semiaxis of the ellipse
% Column 3 b	Length of the vertical semiaxis of the ellipse
% Column 4 x0	 x-coordinate of the center of the ellipse
% Column 5 y0	y-coordinate of the center of the ellipse
% Column 6 phi	Angle (in degrees) between the horizontal semiaxis of the ellipse and the x-axis of the image

[fn, pn] = uiputfile('*.mp4', 'Save Video as');
ffn = fullfile(pn,fn) ;

vid = VideoWriter(ffn, 'MPEG-4') ;
vid.FrameRate = 1/dT ;
open(vid) 

imgt = zeros(n,n,1,nF) ;

for it = 1:nF
    E_this = E;
    E_this(elip,4) = E(elip,4) + 0.5*sin(2*(it-1)/nF*2*pi)/n ;
    imgt(:,:,1,it) = mat2gray(phantom(def,E_this, n),[0 1]) ;
end

writeVideo(vid,imgt) ;
close(vid)

% call video phase processing algorithms
magPhase = 15 ;
fl = 0.05 ;
fh = 0.5 ;
fs = 1/dT ;
outDir = pn ;
params = {'sigma',1, 'pyrType','octave'} ;
[outName, res] = phaseAmplify(ffn, magPhase , fl, fh,fs, outDir, params{:}) ;

eshow(permute(res,[1 2 4 3]),'isrgb',true)




    

