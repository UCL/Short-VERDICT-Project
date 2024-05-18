% Function to generate signal distributions over a scheme given a set of
% tissue parameters

% Signals simulated from Original VERDICT model!

function [signalDists, signals] = SimulateProstateSignalDistributions(scheme, tissue_params, opts)

arguments
    scheme % scheme of scan paramaters

    % tissue parameters
    tissue_params.fIC % Tissue parameters for signal simulation
    tissue_params.DIC = 2
    tissue_params.Rs = linspace(0.1,15.1,50)
    tissue_params.fRs = []
    tissue_params.fEES
    tissue_params.DEES = 2
    tissue_params.fVASC = 0
    tissue_params.DVASC = 8
    tissue_params.T2

    % other options
    opts.simnoisetype = 10000
    opts.sigma = 0.05 % Noise sigma (for TE = 0)
    opts.TEconst = 22 % TE = delta + Delta + TEconst

end

% == Steps

% 1. Calculate mean b=0 signals over scheme
% 2. Calculate mean b\=0 signals over scheme
% 3. Generate ratio distributions over scheme

% ===================================================================


% Default radii distribution if no fRs distribution given
if isempty(tissue_params.fRs)
    tissue_params.fRs = normpdf(tissue_params.Rs, 8, 2);
end

% Normalise and scale fRs
tissue_params.fRs = tissue_params.fIC*tissue_params.fRs/sum(tissue_params.fRs);


% === Bits and bobs

nscheme = length(scheme);
nR = length(tissue_params.Rs);


%% b=0 signals

b0signals = zeros([nscheme 1]) ;

for ischeme = 1:nscheme

    % T2 decay
    TE = scheme(ischeme).TE;

    b0signals(ischeme) = 1*(exp(-TE/tissue_params.T2));

end


%% b\=0 signals

% Empty arrays for signals
sIC = zeros([nscheme nR]) ;
sEES = zeros([nscheme 1]) ;
sVASC = zeros([nscheme 1]) ;
bsignals = zeros([nscheme 1]) ;

for ischeme = 1:nscheme

    % IC signal for different radii
    for ir = 1:nR
        sIC(ischeme,ir) = tissue_params.fRs(ir)*sphereGPD(scheme(ischeme).delta, scheme(ischeme).DELTA, ...
            scheme(ischeme).G, tissue_params.Rs(ir), tissue_params.DIC);
    end

    % EES signal
    sEES(ischeme) = ball(scheme(ischeme).bval, tissue_params.DEES) ;
    
    % VASC signal
    sVASC(ischeme) = astrosticks(scheme(ischeme).bval, tissue_params.DVASC) ;

    % Total signal
    bsignals(ischeme) = sum(sIC(ischeme,:)) + tissue_params.fEES*sEES(ischeme) + ...
        tissue_params.fVASC*sVASC(ischeme);

    bsignals(ischeme) = bsignals(ischeme)*b0signals(ischeme);
    
end


%% Measurement distriutions

% Measurement X = Sb/S0, distribution given by Rice ratio distribution


for ischeme = 1:nscheme

    A0 = b0signals(ischeme);
    Ab = bsignals(ischeme);
    Nav_ratio = scheme(ischeme).Nav_ratio;

    switch opts.simnoisetype

        case 'Rice'
            [signalDist, signals] = RiceDist(A0, Ab, opts.sigma, Nav_ratio = Nav_ratio);

        case 'Ratio'
            [signalDist, signals] = RatioDistRician(A0, Ab, opts.sigma, Nav_ratio = Nav_ratio);
    end
    
    signalDists(ischeme,:) = signalDist;

end