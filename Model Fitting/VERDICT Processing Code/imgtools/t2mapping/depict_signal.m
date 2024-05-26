figure('Name','signal', 'Position', [735 598 441 129])

t = tiledlayout(1,2,'Padding','compact','TileSpacing','compact') ;

te = 0:10:300;
t2sh = 50;
t2l = 300 ;

tem = [30:30:240] ;

ss = exp(-te/t2sh) ;
sl = exp(-te/t2l) ;

ssm = exp(-tem/t2sh) ;
slm = exp(-tem/t2l) ;

lw = 4;
lm= 2 ; % marker linewidth

ct = [0 0 0] ;
cs = [216 174 200]/256 ;
cl = [0 0.8 0.8];

nexttile
lwf = 0.3 ;
plot(te,(1-lwf)*ss, 'LineWidth',lw,'Color',cs), hold on
plot(te,lwf*sl, 'LineWidth',lw,'Color',cl)
plot(te,(lwf*sl + (1-lwf)*ss), 'LineWidth',lw,'Color',ct)
plot(tem,(lwf*slm + (1-lwf)*ssm), 'Linewidth', lm,'Color',ct,'Marker','o', 'MarkerSize',10)
ax = gca ;
ax.YTickLabel = [] ;

ylabel(ax,'Signal')
xlabel(ax,'Echo Time')


nexttile
lwf = 0.1 ;
plot(te,(1-lwf)*ss, 'LineWidth',lw,'Color',cs), hold on
plot(te,lwf*sl, 'LineWidth',lw,'Color',cl)
plot(te,(lwf*sl + (1-lwf)*ss), 'LineWidth',lw,'Color',ct)
plot(tem,(lwf*slm + (1-lwf)*ssm), 'Linewidth', lm,'Color',ct,'Marker','o', 'MarkerSize',10)
ax = gca ;
ax.YTickLabel = [] ;



figure('Name','T2s', 'Position', [735 598 441 129])

t = tiledlayout(1,2,'Padding','compact','TileSpacing','compact') ;

nexttile

t2s = 0:1:500 ;
ps = exp(-(t2sh-t2s).^2/60);
a = area(t2s,ps) ;
hold on
a.FaceColor = cs ;

pl = exp(-(t2l-t2s).^2/40);
a = area(t2s,0.8*pl) ;
a.FaceColor = cl ;
axis([0 500 0 1.5])
ax = gca ;
ax.YTickLabel = [] ;
ax.XTick = 200 ;
ylabel(ax,'P(T2)')
xlabel(ax,'T2')

nexttile
ps = exp(-(t2sh-t2s).^2/40);
a = area(t2s,1.5*ps) ;
hold on
a.FaceColor = cs ;

pl = exp(-(t2l-t2s).^2/40);
a = area(t2s,0.25*pl) ;
a.FaceColor = cl ;
axis([0 500 0 1.5])
ax = gca ;
ax.YTickLabel = [] ;
ax.XTick = 200 ;
