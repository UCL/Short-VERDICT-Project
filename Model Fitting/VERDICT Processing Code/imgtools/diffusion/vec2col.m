function [rgb] = vec2col(vec, anisotropy, min_anis, max_anis)
% VEC2COL Vector (eigenvector direction) to colour
%   rgb = vec2col(vec, anisotropy, min_anis, max_anis) returns an rgb
%         image (or volume) colour coded by eigenvector direction with 
%         intensity related to anisotropy.
%   
%   vec         Matrix of eigenvectors, size N1 x N2 x ... x 3
%   anisotropy  Anisotropy at each point, size N1 x N2 x ...
%   min_anis , max_anis  scalars giving the min and max range of 
%     anisotropies plotted, values outside are black.
%
%   vec is XYZ ie [1 0 0] is red.
%
%   rgb is same size as vec
%
% Example rgb = vec2col(eigvec, fa, 0, 1) ;
%
% based on the paper by Pajevic and Pierpaoli  MRM 42 (3) p526. (1999)
%
% heurisitic parameters are difficult to set.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% See also colwheel
%
% $Id: vec2col.m 163 2007-08-30 20:04:34Z ucacdat $

X=2;Y=1;Z=3;
XC = 1; YC = 2 ; ZC = 3 ;

pe = 1 ;  % 0, RGB equivalent, 1 uniform hues
pb = 0 ;  % decreases the saturation of blue
pr = 0. ;  % similarly for red

if pe > 0 & pb > 0.5/pe ; warning([' pb should not exceed 0.5/pe ']); end

pc = 0.2 ;  % 0 maximal range of colour pb,pe,LE disabled

if (pb*pc)>1 ; warning(['pc*pb too high']); end

%beta = 0.4 ;
%LE = 0.6 ; %reference brightness  for large vals set pr larger

LEbeta = 0.7^(1/0.4) ;

pbeta = 1 ; % non-linear scaling of anisotropy
gamma = 1 ;   % 1.3 or 2.4 for CRT

szv = size(vec) ;
if szv(end)~=3
  error(['Last dimension of vec must be size 3'])
end

vec = reshape(vec, [prod(szv(1:end-1)) 3]) ;

anisotropy = anisotropy(:) ;

jlo = find(anisotropy < min_anis) ;
jhigh = find(anisotropy > max_anis) ;

anis = (anisotropy - min_anis) ./(max_anis - min_anis) ;
anis(jlo) = 0 ;
anis(jhigh) = 0 ;

% remove rounding errors that make input vecs slightly greater than 1
acc = 1000 ;

RI = floor(acc*abs(vec(:,XC))) / acc ;
GI = floor(acc*abs(vec(:,YC))) / acc ;
BI = floor(acc*abs(vec(:,ZC))) / acc ;

if max(RI)>1 ; error(['RI max > 1 : ', num2str(max(RI)) ]); end
if max(GI)>1 ; error(['GI max > 1 ', num2str(max(GI))]); end
if max(BI)>1 ; error(['BI max > 1 ', num2str(max(BI))]); end

% shift the blue towards white
b = BI ./ (RI + GI + BI) ;

CB = 3/2*pb*(b-1/3)*pc ;

% threshold CB to be 0 or higher
CB(find(CB<0)) = 0 ;

RS = CB .* BI + (1-CB).*RI ;
GS = CB .* BI + (1-CB).*GI ;
BS = BI ;

% shift the red towards white

r = RS./(RS+GS+BS) ;
CR = 3/2*pr*(r-1/3)*pc ;

CR(find(CR<0)) = 0 ;

RT = RS ;
GT = CR.*RS +(1-CR).*GS ;
BT = CR.*RS +(1-CR).*BS ;

RS = RT;
GS = GT;
BS = BT ;


fmax = 1.0 ;

c1 = 1/3 - pe/25 ;
c2 = 1/3 + pe/4 ;
c3 = 1-c1-c2;

FL = (c1.*RS + c2.*GS + c3.*BS) ./ LEbeta ;
FL(find(FL>1))= 1 ;

LM = max(max(RS,GS),BS) ;   % typo in paper here?

LF = pc.*FL + (1-pc).*LM ;

FR = fmax * (anis.^pbeta .* (RS./LF)).^(1/gamma) ;
FG = fmax * (anis.^pbeta .* (GS./LF)).^(1/gamma) ;
FB = fmax * (anis.^pbeta .* (BS./LF)).^(1/gamma) ;

% above don't seem to be correctly normalised

FR(find(FR>1))=1 ;
FG(find(FG>1))=1 ;
FB(find(FB>1))=1 ;

%FR = FR./(max(max(max(max(FR))))) ;
%FG = FG./(max(max(max(max(FG))))) ;
%FB = FB./(max(max(max(max(FB))))) ;


dout(:,1) = FR ; % was X 15/02/2005
dout(:,2) = FG ;
dout(:,3) = FB ;

%dout(:,X) = RI ;
%dout(:,Y) = GI ;
%dout(:,Z) = BI ;

rgb = reshape(dout, szv) ;
