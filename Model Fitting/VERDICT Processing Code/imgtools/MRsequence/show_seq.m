function varargout = show_seq(scheme, varargin)
% SHOW_SEQ Displays calculated signal for sequence
%
% show_seq(scheme)
% show_seq(scheme, param, value, ...)
% aF0 = show_seq(...) outputs computed signal
%
% scheme is a string understood by build_seq e.g. 'ssSPGR'
% D.Atkinson@ucl.ac.uk
%
% See also BUILD_SEQ SQ_EPG_GRE
%


% defaults
T1_change_fac = 0.7 ; % T1 reduction factor (for Gd)
T1_change_time = 5000 ;
plot_reg_x = false ; % plot at regular x points (not actual time) - useful 
                     % when comparing to 1Dtestmode data
plot_Zn = false ; 

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    switch varargin{ipv}
        case 'T1_change_fac'
            T1_change_fac = val ;
        case 'T1_change_time'
            T1_change_time = val ;
        case 'plot_reg_x' 
            plot_reg_x = val ;
        case 'plot_Zn' 
            plot_Zn = val ;
        otherwise
            error(['Unknown parameter: ',varargin{ipv}])
    end
end

[sq, series] = build_seq(scheme) ;


for isq = 1:length(sq)
    if sq(isq).tstart > T1_change_time 
        sq(isq).T1 = sq(isq).T1 * T1_change_fac ;
    end
end


if isfield(series, 'kmax')
   [F0,Fn,Zn,F] = sq_epg_gre(sq, 'kmax', series.kmax);
else
   [F0,Fn,Zn,F] = sq_epg_gre(sq);
end

figure('Name',['(show_seq) ',scheme])
times = [sq.tstart] ;

ADCs = [sq.ADCon] ;
yADC = -0.0*ones([1 length(ADCs)]) ; % y-position of dots representing ADC on
loc = find(ADCs == false) ;
tsa = times ;
tsa(loc) = [] ; yADC(loc) = [] ;

if plot_reg_x == false
  plot(tsa,yADC,'r.', 'DisplayName','ADC on')
end
grid on, hold on

aF0 = abs(F0) ; % signal
aF0(loc) = [] ; % remove for times with no acquisition
if nargout > 0
    varargout{1} = aF0;
end

if plot_reg_x
    plot(aF0,'DisplayName',scheme)
    xlabel('Sequential ADC')
    ernst_end_time = length(aF0) ;
else
    plot(tsa, aF0,'DisplayName',scheme)
    xlabel('Sequence time (ms)')
    ernst_end_time = times(end) ;
end

ssSI = ssSPGR(d2r(series.FA), series.TR, series.T1) ;
ssSIred = ssSPGR(d2r(series.FA), series.TR, T1_change_fac*series.T1) ;


plot([0 ernst_end_time],[ssSI ssSI], 'DisplayName', ...
    ['Ernst. FA: ',num2str(series.FA), ' TR: ',num2str(series.TR), ...
    ' T1: ',num2str(series.T1)] )

plot([0 ernst_end_time],[ssSIred ssSIred], '--','DisplayName', ...
    ['Ernst. FA: ',num2str(series.FA), ' TR: ',num2str(series.TR), ...
    ' T1: ',num2str(T1_change_fac*series.T1)] )
legend
axis([0 ernst_end_time 0 0.1])

if plot_Zn
    figure('Name',['Zn'])
    plot(squeeze(Zn(1,:,1)))
    xlabel('RF pulse number')
    ylabel('$$\tilde{Z}_0^{a,b} / M_0$$','interpreter','latex')
    grid on
end



