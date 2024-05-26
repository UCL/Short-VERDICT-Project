function lwi_offmono(S, TEs)

Slog = log(S) ;
figure
plot(TEs, Slog), hold on

[p,S, mu] = polyfit(TEs(1:4),Slog(1:4),1) ;
x1 = [0:5:200];
[y1, delta] = polyval(p,x1,S, mu);
plot(x1, y1,'r-'), hold on
plot(x1, y1-2*delta,'m--', x1, y1+2*delta,'m--')
c = polyval(p,0,S, mu) ;
m = polyval(p,1,S, mu) - c  ;

% Fit to last points
[pl,Sl, mul] = polyfit(TEs(end-3:end),Slog(end-3:end),1) ;
x1 = [0:5:200];
[y1l, deltal] = polyval(pl,x1,Sl, mul);
plot(x1, y1l,'b-'), hold on
plot(x1, y1l-2*deltal,'g--', x1, y1l+2*deltal,'g--')
cl = polyval(pl,0,Sl, mul) ;
ml = polyval(pl,1,Sl, mul) - cl  ;


title(['normr: ',num2str(S.normr),' exp(c): ', num2str(exp(c)),...
    ' -1/slope: ',num2str(-1/m),' end slope: ',num2str(-1/ml)])
axis([0 200 -5 0]), grid on
delta
p
S
mu
