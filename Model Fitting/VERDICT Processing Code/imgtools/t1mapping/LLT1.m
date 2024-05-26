function T1 = LLT1(Mt, itimes)
% LLT1 Look-Locker T1.
% lsqnonlin fit to magnitude data Mt at inversion times itimes
% T1 = LLT1(Mt, itimes)
%
% See R. Deichmann and A Haase. J Magn Reson 96, 608-612 (1992)
% "Quantification of Tl Values by SNAPSHOT-FLASH NMR Imaging"
% M(t) = A - B exp(-t/T1*)
% T1 = T1* (B/A-1) 
%
% Example:
%  dinfo = datparse ;
%  [vol, matp] = d2mat(dinfo,{'ctdt'},'op','fp') ; % Look-Locker
%  T1 = LLT1(vol,matp.ctdtVec) ;
%  T1display(T1)
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also T1DISPLAY D2MAT DATPARSE

% unknowns x = [t1* A B]
thresh = 15 ; % threshold for T1 calc

itimes = itimes(:) ; % column vector
Mt = double(Mt) ;
nt = length(itimes) ;
nd = ndims(Mt) ;
if size(Mt,nd) ~= nt
    error(['Number of times and last dimension of Mt must agree'])
end

sz_Mt = size(Mt) ;

np = prod(sz_Mt(1:end-1)) ;
T1 = zeros([np 1]) ;


Mt = reshape(Mt,[np nt]) ;
Mts = sum(Mt,2) ;
loc = find(Mts > max(Mts(:))/thresh) ;
disp([num2str(length(loc)),' of ',num2str(np),' pixels and ',num2str(nt), ...
    ' and time points.'])

options = optimset('Display','off') ;
hw = waitbar(0,'Calculating T1s') ;
upint = round(length(loc)/100) ;
for il = 1:length(loc)
    if rem(il,upint)==0
      waitbar(il/length(loc),hw) ;
    end
    ip = loc(il) ;
    % starting estimate
    [mint, lmin] = min(Mt(ip,:)) ;
    x0(1) = itimes(lmin) ; % T1 estimated at zero crossing
    x0(2) = Mt(ip,nt) ;
    x0(3) = x0(2) + Mt(ip,1) ; % use + here as M(t) at start is really negative
    
    f = @(x)llcfun(x,itimes,Mt(ip,:)) ;
    
    [x,resnorm,residual,exitflag] = lsqnonlin(f,x0,[0 0 0],[],options) ;
    T1(ip) = x(1)*(x(3)/x(2) - 1) ;
    
end
close(hw) ;

T1 = reshape(T1,[sz_Mt(1:end-1)]) ;


%
function cf = llcfun(x,times, Mt)
% LLCFN Look-Locker cost function for lsqnonlin

cf = abs(x(2) - x(3).*exp(-times./x(1)))  - Mt(:) ;
