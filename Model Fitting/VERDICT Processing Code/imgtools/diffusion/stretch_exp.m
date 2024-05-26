function [S0, D, a, ADC1, S01] = stretch_exp(DWI, bv)
% STRETCH_EXP Fit a stretched exponential to diffusion data
% Pixelwise fit using lsqnonlin - slow for volumes. 
%  
%  [S0, D, a, ADC1, S01] = stretch_exp(DWI, bv)
%
% DWI is size [ n1 n2 ... nbv]
% bv are the nbv b-values
%
% S0, D and a. Fitted to DWI = S0 exp( -(bv.D)^a )
% ADC1 and S01 are from a mono-exponential fit to all the data.
%
%
% Example:
%  dinfo = datparse ;
%  [DWI,matp] = d2mat(dinfo,{'slice','bv'},'op','dv') ;
%  [S0, D, a, ADC1, S01] = stretch_exp(DWI,matp.bvVec) ;
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also  D2MAT DATPARSE calcADC

% unknowns x = [S0 D a]
thresh = 15 ; % threshold for calc

bv = bv(:) ; % column vector
DWI = double(DWI) ;
nbv = length(bv) ;
nd = ndims(DWI) ;
if size(DWI,nd) ~= nbv
    error(['Number of b-values and last dimension of DWI must agree'])
end

[ADC1, S01] = calcADC(DWI, bv) ;  % mono-exponential fit


sz_DWI = size(DWI) ;

np = prod(sz_DWI(1:end-1)) ; % no. pixels
D = zeros([np 1]) ; 
S0 = zeros([np 1]) ;
a = zeros([np 1]) ;

DWI = reshape(DWI,[np nbv]) ;

DWIs = sum(DWI,2) ;
loc = find(DWIs > max(DWIs(:))/thresh) ;
disp([num2str(length(loc)),' of ',num2str(np),' pixels ',num2str(nbv), ...
    ' and b-values.'])

options = optimset('Display','off') ;
hw = waitbar(0,'Calculating diffusion parameters') ;
wbupdate = round(length(loc)/100) ;

for il = 1:length(loc)
    if rem(il,wbupdate) == 0
        waitbar(il/length(loc),hw) ;
    end
    ip = loc(il) ;
    % starting estimate x0=[S0 D a]
    x0(1) = S01(ip) ;
    x0(2) = ADC1(ip) ;
    x0(3) = 1 ;
    
    f = @(x)secfun(x,bv,DWI(ip,:)) ;
    
    [x,resnorm,residual,exitflag] = lsqnonlin(f,x0,[],[],options) ;
    S0(ip) = x(1) ;
    D(ip) = x(2) ;
    a(ip) = x(3) ;
    
end
close(hw) ;

S0 = reshape(S0,[sz_DWI(1:end-1)]) ;
D = reshape(D,[sz_DWI(1:end-1)]) ;
a = reshape(a,[sz_DWI(1:end-1)]) ;


%
function cf = secfun(x,bv, DWI)
% Stretched exponential cost function
cf = (x(1)*exp( -(bv*x(2)).^x(3)))  - DWI(:) ;
