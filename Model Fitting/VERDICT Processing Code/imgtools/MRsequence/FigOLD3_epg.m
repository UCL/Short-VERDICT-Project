function Fig3_epg
% Figure 3 showing EPG for single shot and SPAIR interrupted 
%
% Assume instantaneous change in T1 at time 5s.
%
%
% D.Atkinson@ucl.ac.uk
%
% See also SHOW_SEQ

schemes = { 'SPAIRSPGRnosweep' , 'ssSPGR' };

hf = figure('DefaultAxesFontSize',10,...
      'DefaultAxesFontWeight', 'bold', ...
    'DefaultAxesLineWidth',2, ...
    'Units','centimeters') ;
lw = 2 ; % plot linewidth


for ischeme = 1:length(schemes)
    [sq, series] = build_seq(schemes{ischeme}) ;
    
    % adjust T1 to be 900ms after 5s of sequence
    T1_fac = 900 / series.T1 ;
    
    for isq = 1:length(sq)
        if sq(isq).tstart > 5000
            sq(isq).T1 =  sq(isq).T1 * T1_fac ;
        end
    end
    
    
    if isfield(series, 'kmax')
        [F0,Fn,Zn,F] = sq_epg_gre(sq, 'kmax', series.kmax);
    else
        [F0,Fn,Zn,F] = sq_epg_gre(sq);
    end
    
    
    times = [sq.tstart] ;
    
    ADCs = [sq.ADCon] ;
    yADC = -0.0*ones([1 length(ADCs)]) ; % y-position of dots representing ADC on
    loc = find(ADCs == false) ;
    tsa = times ;
    tsa(loc) = [] ; yADC(loc) = [] ;
    
    
    % plot(tsa,yADC,'r.', 'DisplayName','ADC on')
    grid on, hold on
    
    aF0 = abs(F0) ;
    aF0(loc) = [] ;
    plot(tsa, aF0, 'LineWidth',lw,'DisplayName',schemes{ischeme})
    xlabel('Sequence time (ms)')
end

ssSI = ssSPGR(d2r(series.FA), series.TR, series.T1) ;
ssSIred = ssSPGR(d2r(series.FA), series.TR, T1_fac*series.T1) ;


plot([0 times(end)],[ssSI ssSI], 'LineWidth',lw, 'DisplayName', ...
    ['Ernst. FA: ',num2str(series.FA), ' TR: ',num2str(series.TR), ...
    ' T1: ',num2str(series.T1)] )

plot([0 times(end)],[ssSIred ssSIred], '--','LineWidth',lw, 'DisplayName', ...
    ['Ernst. FA: ',num2str(series.FA), ' TR: ',num2str(series.TR), ...
    ' T1: ',num2str(T1_fac*series.T1)] )
legend
axis([0 10000 0 0.1])

