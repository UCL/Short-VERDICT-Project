function t1fit_check(r,c,s,M,FAs, TRs, B1, T1_plot, M0_scale)
% T1FIT_CHECK

szM = size(M) ;

if nargin < 7 | isempty(B1)
    B1 = 100*ones(szM(1:end-1)) ;
end
[fit] = MFAT1(M(r,c,s,:), FAs, TRs, B1(r,c,s),'x0_meth','linear', ...
    'optdisplay','iter') ;

    
TRs = double(TRs) ;

figure
plot(FAs*B1(r,c,s)/100, squeeze(M(r,c,s,:)),'ro-')
hold on
% SI = M .* sin(FAs_rad).*((1-E)./(1-cos(FAs_rad).*E));
xFAs = [0:0.5:30] ;
xFAs_rad = xFAs * 2* pi/360 ;

E = exp(-TRs/(fit.T1_lin)) ;
SI_lin = fit.M0_lin*sin(xFAs_rad).*((1-E)./(1-cos(xFAs_rad).*E)) ;

Enl = exp(-TRs/(fit.T1_nl)) ;

SI_nl = fit.M0_nl*sin(xFAs_rad).*((1-Enl)./(1-cos(xFAs_rad).*Enl)) ;

plot(xFAs, SI_lin,'b')
plot(xFAs, SI_nl, 'k')


if exist('T1_plot','var')
    Ep = exp(-TRs/T1_plot) ;
    if ~exist('M0_scale','var')
        M0_scale = 1 ;
    end
    SI_plot = M0_scale*fit.M0_lin*sin(xFAs_rad).*((1-Ep)./(1-cos(xFAs_rad).*Ep)) ;
    plot(xFAs, SI_plot, 'g')

    legend('Data','Linear','Non-linear',['T1: ',num2str(T1_plot)])
else
    legend('Data','Linear','Non-linear')
end
grid

title(['T1_lin = ',num2str(fit.T1_lin),'  non-lin: ',num2str(fit.T1_nl)])


T1 = [150:10:2400] ;
M0 = linspace(0, fit.M0_lin*2,100)  ;
cf = zeros([length(T1) length(M0)]) ;

FAs_rad = B1(r,c,s)/100*FAs*2*pi/360;
TRs = repmat(TRs,[1 length(FAs)]) ;

for it1 = 1:length(T1)
    for im0 = 1:length(M0)
        E = exp(-TRs/T1(it1)) ;
        SI = M0(im0)*sin(FAs_rad).*((1-E)./(1-cos(FAs_rad).*E)) ;
        res = norm(squeeze(M(r,c,s,:))' - SI) ;
        cf(it1,im0) = res ;
    end
end
[cmin,imin] = min(cf(:)) ;
[mint,minm] = find(cf==cmin) ;



figure
contour(M0,T1,cf,60)
xlabel('M0')
ylabel('T1')
grid
hold on
plot(fit.M0_lin, fit.T1_lin,'bo')
plot(fit.M0_nl, fit.T1_nl,'ko')
plot(M0(minm),T1(mint),'ro')
title('Blue linear, black non-lin, red global')


        
    
