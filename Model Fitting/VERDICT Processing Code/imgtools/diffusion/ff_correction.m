function ff_correction(varargin)
% FF_CORRECTION Correction of ADC for Fat Fraction
%
% ff_correction  uses defaults
% ff_correction('param', value, ...)
%
% Available parameters with {default values} are:
%  'ff'  {0.1}   fraction of fat in tissue
%  'alpha' {0.1} fraction of unsupressed fat
%  'TE'  {80}    TE for sequence (ms)
%  'T2w' {22.5}  T2 of water in tissue (ms)
%  'T2f' {72}    T2 of fat (ms)
%
% Example:
%   ff_correction('ff', 0.14, 'alpha', 0.11)
%
% From Dieckmeyer et al MRM 2016
%
% D.Atkinson@ucl.ac.uk 
%

% Default values
def_alpha = 0.1 ;  % Fraction of unsupressed fat.

def_ff = 0.1 ; % Fat fraction

def_TE = 80 ;     % TE - same for both b-values
def_T2w = 22.5 ;  % Water T2 (here in bone)
def_T2f = 72 ;    % Fat T2

% b-values of scan
bvals = [50 900] ; 

% Check input parameter value pairs and set values
p = inputParser ;
addParameter(p,'alpha', def_alpha) ;
addParameter(p,'ff', def_ff) ;
addParameter(p,'TE', def_TE);
addParameter(p,'T2w', def_T2w) ;
addParameter(p,'T2f', def_T2f) ;
parse(p,varargin{:}) ;

alpha = p.Results.alpha ; 
ff = p.Results.ff ;
TE = p.Results.TE ;
T2w = p.Results.T2w ;
T2f = p.Results.T2f ;


ADC2s = [100 : 100 : 2000]*1e-6 ; % ADC range for first plot
indADC = 10 ; % index into above array for second plot

ADC1s = zeros(size(ADC2s)) ;
S0s = zeros(size(ADC2s)) ;

for iadc = 1:length(ADC1s)
    
    % Dieckmeyer model 2
    Smodel = alpha * ff * exp(-TE/T2f) + ...
        (1 - ff) .* exp(-TE/T2w) .* exp(-bvals .* ADC2s(iadc)) ;
    
    % Standard ADC calc from fitting a line to log signal, based on 2
    % b-values.
    
    % S = S0 exp(-b.ADC1)
    % ln(S) = -b.ADC1  +ln(S0)
    % y1 = m.x1 + c
    % y2 = m.x2 + c
    % y1 - y2 = m(x1 - x2)
    % m = (y1-y2)/(x1-x2)
    
    ADC1s(iadc) = (log(Smodel(1)) - log(Smodel(2))) / (-bvals(1) + bvals(2)) ;
    lgS0 = log(Smodel(1)) + bvals(1)*ADC1s(iadc) ;
    S0s(iadc) = exp(lgS0) ;
    
end

% Plot ADCs
figure
plot(ADC1s*1e6, ADC2s*1e6)
xlabel('ADC1s Mono-exp fit')
ylabel('ADC2s Water component')
hold on, grid on
plot(ADC2s*1e6, ADC2s*1e6,'r--')
title(['FF: ',num2str(ff),', alpha: ',num2str(alpha)])

% Plot Component Signals
figure
bvp = [0:10:2000] ; % b-values for plotting
S1 = S0s(indADC)* exp(-bvp*ADC1s(indADC)) ; % Mono-exp
plot(bvp,S1)
title(['ADC1: ',num2str(ADC1s(indADC)*1e6),', ADC2: ',num2str(ADC2s(indADC)*1e6), ...
    ', FF: ',num2str(ff),', alpha: ',num2str(alpha)])

hold on, grid on

S2f =  alpha * ff * exp(-TE/T2f) ; % fat component (constant with b-value)
S2w =  (1 - ff) .* exp(-TE/T2w) .* exp(-bvp .* ADC2s(indADC)) ; % water component
S2 = S2f + S2w ;

plot(bvp,S2w)
plot([min(bvp) max(bvp)],[S2f S2f])
plot(bvp, S2)
xlabel('b-value')
ylabel('Signal')
legend('Mono-exp (ADC1)','water (ADC2)','fat','water + fat')

    





