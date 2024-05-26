function IRT1_check(Mt, itimes, fit, t1_manual) 
% IRT1_check Inversion Recovery T1 calculation check.
% lsqnonlin fit to magnitude data Mt at inversion times itimes
% IRT1_check(Mt, itimes, fit)
% IRT1_check(Mt, itimes, fit, t1_manual)
%
% Example (MultiFrame)
%  dinfoIR = datparse(dselect) ;  % select  Inversion Recovery.
%  ITYPE = 6 ;  % 6 for magnitude, 8 for real 
%  [vir, mir] = d2mat(dinfoIR,{'InversionTime','itype'},'itype',ITYPE,'op','fp') ; 
%  [T1, M0] = IRT1(vir, mir.tiVec,ITYPE) ;
%
% Example (SingleFrame)
%  dinfo = datparse ;
%  ITYPE = 6 ;  % 6 for magnitude, 8 for real 
%  [vir, mir] = d2mat(dinfo,{'series','itype'},'series', [ ], 'itype',ITYPE,'op','fp') ; 
%  [T1, M0] = IRT1(vir, mir.tiVec_indata,ITYPE) ;
%
%  T1display(T1)
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also LLT1 T1DISPLAY D2MAT DATPARSE IRT1


thresh = 15 ; % threshold for T1 calc

itimes = itimes(:) ; % column vector
Mt = double(Mt) ;
nt = length(itimes) ;
nd = ndims(Mt) ;
if size(Mt,nd) ~= nt
    error(['Number of times and last dimension of Mt must agree'])
end

% if itype ~= 6 && itype ~= 8
%     error(['itype must be 6 or 8'])
% end
%  
% Added for checking
hf = figure('Name','Click on figure') ;
h_im = imshow(fit.T1,[0 3000]) ;
while 1> 0
imp = impoint(gca) ;
posc = getPosition(imp) ;
posc = round(posc) ;
r = posc(2) ;
c = posc(1) ;

%calct = linspace(0,max(itimes),200)' ;
calct = linspace(0,15000,200)' ;
if nargin < 5
    t1_manual = [] ;
end

switch fit.opt
    case 'lsqnl_mag2'
        X = [fit.T1(r,c), fit.M0(r,c) ];
    case 'lsqnl_mag3'
        X = [fit.T1(r,c), fit.M0(r,c), fit.RB(r,c)] ;
    case 'RD-NLS-PR3'
        X = [fit.T1(r,c), fit.M0(r,c), fit.RB(r,c)/(-2*fit.M0(r,c))] ;
end
ir = ircfun(X , calct, fit.opt) ; % Mt=0 here so ir is just computed recovery

cf = ircfun(X, itimes, fit.opt, Mt(r,c,:)) ;

figure('Name','IRT1_check')
plot(calct,ir), hold on, grid on
plot(itimes,squeeze(Mt(r,c,:)),'-o')
if ~isempty(t1_manual)
    X(1) = t1_manual ;
    irm = ircfun(X , calct, fit.opt) ;
    plot(calct,irm)
end
switch fit.opt
    case 'RD-NLS-PR3'
        PR = ones(size(itimes)) ;
        PR(1:fit.tau(r,c)) = -1 ;
        plot(itimes,PR.*squeeze(Mt(r,c,:)),'-o')
end
xlabel('Time')
title(['T1: ',num2str(fit.T1(r,c)), ...
    ' norm(cf) ',num2str(norm(cf)), ' (r: ',num2str(r),' c: ',num2str(c),')'])

figure(hf)
end % while


%
% function cf = irfun(x,times, itype)
% % IRFN Inverstion Recovery function 
% % See also LLCFN 
% 
% if itype == 6
%    cf = abs(x(2) * (1 - 2.*exp(-times./x(1))))   ; 
% elseif itype == 8
%    cf = (x(2) * (1 - 2.*exp(-times./x(1))))  ; 
% else
%     error
% end
