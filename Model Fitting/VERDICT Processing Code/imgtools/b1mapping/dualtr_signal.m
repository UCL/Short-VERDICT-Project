function dualtr_signal(TR1, TR2, T1)
% DUALTR_SIGNAL Predicts signal from a dual TR acquisition for B1 mapping
%
% dualtr_signal(TR1, TR2, T1)
%
% From Yarnykh MRM 57p192 (2007)
%

fa = [0:1:100] ;

fa_rad = fa*2*pi/360 ;

E1 = exp(-TR1/T1) ;
E2 = exp(-TR2/T1) ;
cfa = cos(fa_rad);

M1 = ( 1- E2 + (1-E1)*E2.*cfa ) ./ ...
    (1 - E1*E2.*cfa.*cfa) ;

M2 = ( 1- E1 + (1-E2)*E1.*cfa ) ./ ...
    (1 - E1*E2.*cfa.*cfa) ;

S1 = M1 .* sin(fa_rad) ;
S2 = M2 .* sin(fa_rad) ;

r = S2./S1 ;
n = TR2/TR1 ;

fa_comp = 360/2/pi*acos((r.*n - 1) ./ (n-r)) ;

% figure
% plot(fa,fa_comp), grid on


figure
plot(fa,S1), hold on, grid on
plot(fa,S2)
title(['T1: ',num2str(T1)])
legend([num2str(TR1),' ms'],[num2str(TR2),' ms'])