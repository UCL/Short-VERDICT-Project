function [CBV, FM] = gv_fit(C, bend, bolus, plotgv)
% GV_FIT Gamma variate fit for DSC data using Madsen method
%
%  [CBV, FM] = gv_fit(C, bend, bolus) 
%  [CBV, FM] = gv_fit(C, bend, bolus, plotgvTF) 
%
% C [ndyn]
% bend index of last point of baseline data in C
% bolus no. of dynamics in bolus (to prevent fitting to later time points)
%  defaults to 4
%
% Masden Phys. Med. Bid., 1992, Vol. 31, No 1, 1597-1600.
%
% David Atkinson
%
NP = 400 ; % no. of points for gamma variate plot and CBV integration

if nargin < 3
  bolus = 4 ;
end
if nargin< 4
    plotgv = false ;
end

ndyn = length(C) ;

jbolus = [bend: bend+1+bolus] ; % points for fit
if jbolus(end) >= ndyn
    CBV= 0 ; FM = 0 ;
    return
end

tt = [1:ndyn]; 
t0 = tt(bend) ; % Define t0 as end of baseline, next point should be 
                % where we see contrast

%estimate ymax, tmax from peak
[ymax_est, itmax] = max(C) ;
tmax_est_peak = tt(itmax) ;

% estimate from centroid of peak NOT USED
tmax_est_cent = sum(tt(jbolus).*C(jbolus)) / sum(C(jbolus)) ;

% disp(['tmax_est_centroid: ',num2str(tmax_est_cent),' peak: ',num2str(tmax_est_peak)])

tmax = tmax_est_peak ;

tp = (tt - t0) / (tmax - t0) ;

X = 1 + log(tp) - tp ;
Y = log(C); 


Xr = real(X) ;
Yr = real(Y) ;
P = polyfit(Xr(bend+1: bend+1+bolus),Yr(bend+1: bend+1+bolus),1) ;

% figure
% plot(Xr(bend+1:end),Y(bend+1:end)), hold on
% plot([Xr(bend+1) Xr(bend+1+bolus)], [P(2) + P(1)*Xr(bend+1)  P(2) + P(1)*Xr(bend+1+bolus)])

alpha = P(1) ;
ymax = exp(P(2)) ;

A = ymax * (tmax-t0).^(-alpha) * exp(alpha) ;
beta = (tmax-t0) / alpha ;

% First Moment from Christensen
% JOURNAL OF MAGNETIC RESONANCE IMAGING 27:1371–1381 (2008)

FM = t0+ beta*(alpha+1) ;

ttr = tt - t0 ;

ttr_p = linspace(min(ttr),max(ttr),NP) ;
yp = zeros([1 NP]);

yp = A * (ttr_p.^alpha) .* exp(-ttr_p/beta) ;

loc = ttr_p<0 ;
yp(loc) = 0 ;

CBV = trapz(yp) * (ttr_p(2)-ttr_p(1)) ;

if plotgv
    figure('Name','Gamma variate fit')
    plot(tt,C), hold on
    plot(ttr_p+t0,yp)
    plot([FM FM], [0 max(C)/10])
    title(['CBV: ',num2str(CBV),'. First Moment: ',num2str(FM)])
end




