function entropy_plot

p = [0.001:0.01:1.2] ;

E = -p.*log(p) ;

figure('Units','pixels','Position',[100 100 500 400])
plot(p,E)
grid on
xlabel('p')
ylabel('-p.log(p)')
title('-p.log(p) vs p')
