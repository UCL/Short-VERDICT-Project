function conc2SI_ssSPGR(FA_deg, TR)
C = [0:0.005:2] ; % mM
T10 = 1.400 ; % s
r1 = 4 ; % Mm-1 s-1

R = 1/T10 + r1.*C ;

T1s = 1000./ R ; % ms

hf = figure('DefaultAxesFontSize',10,...
      'DefaultAxesFontWeight', 'bold', ...
    'DefaultAxesLineWidth',2, ...
    'Units','centimeters')
 
lw = 2 ; % plot linewidth
 
yyaxis left


for iperm = 1:length(FA_deg)
    FA_rad = d2r(FA_deg(iperm)) ;

    SI = ssSPGR(FA_rad, TR(iperm), T1s) ;

    dSIdC = (SI(2)-SI(1))/(C(2)-C(1)) ;
    plot(C,SI,'LineWidth',lw, 'DisplayName',['FA: ',num2str(FA_deg(iperm)), ...
        ' TR: ',num2str(TR(iperm)), ' slope: ',num2str(dSIdC)])
    hold on
    SI0 = SI(1) ;
    [mn, loc] = min(abs(SI-SI0*1.5)) ;
    plot(C(loc),SI(loc),'o')
    
end

text( 0.5, 0.07,'\bf\fontsize{12}5ms, 15\circ')
text( 1.4, 0.11,'\bf\fontsize{12}5ms, 10\circ')
text( 0.7, 0.12,'\bf\fontsize{12}\leftarrow 10 ms, 10\circ')
text( 0.45, 0.15,'\bf\fontsize{12}10 ms, 15\circ\rightarrow')

grid on
axis([0 min([2 max(C)]) 0 0.2])

xlabel('C [mM]', 'FontWeight','bold')
ylabel('Signal Intensity', 'FontWeight','bold')

yyaxis right
plot(C,T1s, 'LineWidth',lw, 'DisplayName','T1 vs conc')
text(1.5, 200,'T1','FontWeight','bold', 'FontSize',12)
ylabel('T1 (ms)', 'FontSize', 10, 'FontWeight','bold')
%legend('location','best')

% Plot SI vs FA for representative TRs and [CA]
figure('Name','SI vs FA')

CAs = [0 0.01 0.02 0.04 0.08 0.16] ;
TRs = [5 10] ;

nTR = length(TRs) ;
nCA = length(CAs) ;

CAsp = repmat(CAs,[1 nTR]) ;
TRsp = [] ;
for iTR = 1: nTR
   TRsp =  [TRsp repmat(TRs(iTR),[1 nCA]) ];
end


FAs = [0:0.1:50];


R = 1/T10 + r1.*CAsp ;

T1sp = 1000./ R ; % ms

for iperm = 1:length(TRsp)
    FAs_rad = d2r(FAs) ;

    SI = ssSPGR(FAs_rad, TRsp(iperm), T1sp(iperm)) ;

    if iperm <= nCA
      lcol = [1 0 0 ] ;
    else
        lcol = [0 0 1] ;
    end
    
    plot(FAs,SI,'LineWidth',lw, 'Color', lcol, ...
        'DisplayName',['TR: ',num2str(TRsp(iperm)), ...
        ' [CA]: ',num2str(CAsp(iperm)) ])
    hold on
    
end
grid on
legend

%%
figure('Name','SI vs FA')
T10 = 1.4
r1 = 4;

FAs = [0:0.1:50];
FAs_rad = d2r(FAs) ;
TRs = [ 5 10] ;

CAs = [ 0 0.2] ;
R = 1/T10 + r1.*CAs ;

T1s = 1000./ R ; 

for iperm = 1: length(TRs)
    SI0 = ssSPGR(FAs_rad, TRs(iperm), T1s(1)) ;
    SI1 = ssSPGR(FAs_rad, TRs(iperm), T1s(2)) ;
    
    plot(FAs, (SI1-SI0))
    hold on
    grid on
end

%% FAmax vs conc change

CAds = [0.01 : 0.01 : 2] ; 
TRs = [ 5 : 10] ;
T10 = 1.4 ;
r1 = 4;

FAs = [0 : 0.01 : 50 ];
FAs_rad = d2r(FAs) ;

R = 1/T10 + r1.*CAds ;
T1s = 1000./ R ; 

figure 
yyaxis left
for iTR = 1:length(TRs)
    FAmax = zeros([1 length(CAds)]) ;
    SId = zeros([1 length(CAds)]) ;
    for iCA = 1:length(CAds)
      SI = ssSPGR(FAs_rad, TRs(iTR), T1s(iCA)) ;
      SI0 = ssSPGR(FAs_rad, TRs(iTR), T10*1000 ) ;
      dSI = SI - SI0 ;
      
      [maxs, loc] = max(dSI) ;
      SId(iCA) = maxs ;
      FAmax(iCA) = FAs(loc) ;
    end
    yyaxis left
    plot(CAds,FAmax,'DisplayName',[num2str(TRs(iTR))])
    ylabel('FAmax')
    hold on
    axis([0 max(CAds) 0 20])
    yyaxis right
    plot(CAds, SId)
    ylabel('SI')
    xlabel('[CA]')
    grid on
end
legend


