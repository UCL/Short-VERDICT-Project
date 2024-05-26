
lw = 4 ; % linewidth
cbi =     [0.85  0.325  0.098] ;
cnnls   = [0.929  0.694  0.125] ;
cdevine = [0.446  0.674  0.188] ;

pc = {'Position', [735 598 450 200] } ;

mt2 = 800 ;

bifitfn = pref_uigetfile('plotfits','bifitfn') ;
D =load(bifitfn) ;

hfsig = figure('Name','sig','Position',[100 100 400 300]) ;
loc = find(1000*D.TEcalc > 30) ;
plot(1000*D.TEcalc(loc), D.flong(loc)+D.fshort(loc),'Color',cbi,'LineWidth',lw)
hold on
plot(1000*D.TEs, D.sig, 'Marker','o', 'LineWidth',2,'MarkerSize',10,'Color',[0 0 0])


ht2 = figure('Name','t2', pc{:}) ;
stem(1000*D.T2short, D.a(1,D.mloc)/200000, 'filled','Color',cbi, 'LineWidth',lw)
hold on
stem(1000*D.T2long, D.c(1,D.mloc)/200000, 'filled','Color',cbi , 'LineWidth',lw)



disp('Select nnls')
nnlsfitfn = pref_uigetfile('plotfits','nnlsfitfn') ;
D =load(nnlsfitfn) ;

figure(hfsig)
loc = find(1000*D.TEcalc > 30) ;
plot(1000*D.TEcalc(loc), D.sig(loc), 'LineWidth',lw, 'Color',cnnls,  'LineStyle','--')

figure(ht2)
plot(1000*D.T2T, D.T2D/max(D.T2D(:)), 'Color',cnnls, 'LineWidth',lw, 'LineStyle','--')



disp('Select Devine')
devinefitfn = pref_uigetfile('plotfits','devinefitfn') ;
D =load(devinefitfn) ;

figure(hfsig)
loc = find(1000*D.TEcalc > 30) ;
plot(1000*D.TEcalc(loc), D.C(loc),'Color',cdevine, 'LineWidth',lw)
xlabel('Echo Time (ms)','FontSize',14)
ylabel('Signal','FontSize',14)
ax = gca ;
ax.YTickLabel = [] ;
axis([0 300 0 1.2*max(D.C(:))])


figure(ht2)
plot( 1000*linspace(0,2,1000), D.P/max(D.P(:)), 'Color',cdevine, 'LineWidth', lw)
ax = gca ;
ax.YTickLabel = [] ;
ax.XTick = [0 200 400 600 800] ;
axis([0 mt2 0 1.5])
xlabel('T2 (ms)','FontSize',14)
ylabel('P(T2)','FontSize',14)


