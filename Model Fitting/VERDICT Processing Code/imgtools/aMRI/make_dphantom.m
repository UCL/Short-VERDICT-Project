function imgt = make_dphantom
% MAKE_DPHANTOM Make video of dynamic phantom to trial aMRI 
%
%  imgt = make_dphantom
%    imgt is [n n 1 nF]
%


nF = 100 ; % Number of Frames
n= 128 ; % Size of image

def = 'Modified Shepp-Logan' ; 

[P, E] = phantom(def,n);
elip = 4 ;

% Column 1 A	Additive intensity value of the ellipse
% Column 2 a Length of the horizontal semiaxis of the ellipse
% Column 3 b	Length of the vertical semiaxis of the ellipse
% Column 4 x0	 x-coordinate of the center of the ellipse
% Column 5 y0	y-coordinate of the center of the ellipse
% Column 6 phi	Angle (in degrees) between the horizontal semiaxis of the ellipse and the x-axis of the image

imgt = zeros(n,n,1,nF) ;

for it = 1:nF
    E_this = E;
    E_this(:,4) = E(:,4) + 1*sin(2*(it-1)/nF*2*pi*6)/n  + it/n/5 - nF/n/5/2;
    imgt(:,:,1,it) = mat2gray(phantom(def,E_this, n),[0 1]) ;
end






    

