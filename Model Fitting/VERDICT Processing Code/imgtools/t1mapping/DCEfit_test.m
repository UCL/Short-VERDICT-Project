function DCEfit_test
% Test for DCEfit
%
%

ny = 1 ; nx = 1 ; N = ny*nx ;
Stimes = [0:2:3*60] ;
nt = length(Stimes) ;

Smeas = zeros([ny*nx nt]);


tonset = 0 ;
Ktrans = 0.451 / 60 ; % s-1
ve = 0.701 ;
vp = 0.40 ; % percentage
tonset = 20 ;
x = [ Ktrans ve vp tonset] ;
% model.tonset = 0 ;

rlx = 4.9 ;  % s-1 mM-1

T10fit.T1_nl = 1000.* ones([ny nx]) ;
T10fit.M0_nl = ones([ny nx]) ;
M0 = T10fit.M0_nl ;
T10 = T10fit.T1_nl ;

TR = 4 ;
FA_rad = 10 / 360 * 2 * pi ;



model.AIFmethod = 'Parker';
model.DCEmethod = 'ETM';
DCEfit_true = repmat(x,[ny*nx 1]);

for ip = 1:N
  c = pharma(DCEfit_true(ip,:), Stimes, model) ;
  Smeas(ip,:) = C2S(c ,rlx, T10(ip), M0(ip), FA_rad, TR) ;
end
plot(Stimes,Smeas(1,:))

fit = DCEfit(T10fit, Smeas,Stimes, FA_rad, TR) ;
ipix = 1;
cf = pharma(fit(ipix,:), Stimes, model) ;
Spred(ipix,:) = C2S(cf ,rlx, T10(ipix), M0(ipix), FA_rad, TR) ;

figure
plot(Stimes, Spred(1,:),'r'), hold on
plot(Stimes, Smeas(1,:),'k')

disp(['True Ktrans: ',num2str(60*DCEfit_true(1,1)),', ve: ',num2str(DCEfit_true(1,2)), ...
    ', vp: ',num2str(DCEfit_true(1,3))])

disp(['Fitted Ktrans: ',num2str(60*fit(1,1)),', ve: ',num2str(fit(1,2)), ...
    ', vp: ',num2str(fit(1,3))])

if size(fit,2)==4
    disp(['Fitted tonset: ',num2str(fit(1,4))])
end



