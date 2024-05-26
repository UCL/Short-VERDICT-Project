function adaptiveSPGR_EPG
% ADAPTIVESPGR_EPG Extended Phase Graph simulations for modified RF spoil
%
% Uses a single representative TR/FA to avoid confusion.
%
% Also plots Ernst solution as a step
%
% D.Atkinson@ucl.ac.uk
%
% See also cep_doctor ssSPGR sq_epg_gre build_seq
%

% Copyright 2018 University College London.

% Figure and axes properties
hf = figure('Name','EPG', ...
      'DefaultAxesFontSize',12,...
      'DefaultAxesFontWeight', 'bold', ...
      'DefaultAxesLineWidth',2, ...
      'Units','centimeters') ;
lw = 2;

schemes = {  'generic_proset'} ;
lws = { 1} ;
dn = {'single-shot'} ;

T2 = 80 ;
T1 = 1400 ;
T1_change_fac = 1100 / T1 ; % simulated change in T1 due to [Gd].
T1_change_time = 5000 ; 
TR = 7 
FA = 11


for ischeme = 1:length(schemes)
    
    [sq, series] = build_seq(schemes{ischeme},'TR',TR,'FA',FA,'T2',T2,...
        'T1',T1, 'spoil_incr', 150) ;
    
    for isq = 1:length(sq)
        if sq(isq).tstart > T1_change_time  
            sq(isq).T1 =  sq(isq).T1 * T1_change_fac ;
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
    plot(tsa, aF0, 'LineWidth',lws{ischeme},'DisplayName',dn{ischeme})
    xlabel('Sequence time (ms)')
    ylabel('Signal')
end

% plot the Ernst steady-state 
ssSI =    ssSPGR(d2r(series.FA), series.TR, series.T1) ;
ssSIred = ssSPGR(d2r(series.FA), series.TR, T1_change_fac*series.T1) ;

plot([0 T1_change_time T1_change_time+series.TR times(end)], ...
    [ssSI ssSI ssSIred ssSIred], 'LineWidth',lw*2,'Color',[0 0 0], ...
    'LineStyle',':', 'DisplayName', 'Ernst signal')

legend
axis([0 7000 0  0.4])



  