% Function to simulate noisy signals over specified scheme for specified
% model


function signals  = simulateSignalsOverScheme(scheme, modeltype, tissue_params, opts)


arguments

    scheme % Specify scheme
    modeltype % Specify model type

    % == Tissue parameters
    
    % == Core
    tissue_params.fIC
    tissue_params.fEES
    tissue_params.fVASC = 0
    
    % == VERDICT
    tissue_params.fRs = []
    tissue_params.Rs = []

    % == RDI
    tissue_params.fRdists = []
    tissue_params.muRs = []
    tissue_params.sigmaRs = []
    % v1.3 and 1.4
    tissue_params.muR 
    tissue_params.sigmaR


    % == Options

    opts.sigma0 = 0.05;
    opts.T2 = 100;
    opts.TEconst = 22;

    opts.noisetype = 'Ratio';

end


% STEPS

% 1. Generate signal distribution for each scan
% 2. Sample distributions to get signals

% ========================================================================

% Scheme length
nscan = length(scheme);

%% 1. Generate signal distributions for each scan

% Initialise array for signals
signals = zeros(1, nscan);

% Loop over scans
for scanIndx = 1:nscan

    % Check if b=0
    if scheme(scanIndx).bval == 0
        signals(scanIndx) = 1;
        continue
    end

    % Scan parameters
    scan = scheme(scanIndx);
    scan_params = [scan.delta, scan.DELTA, scan.bval];

    % Echo time
    TE = scan.TE; %scan.delta + scan.DELTA + opts.TEconst;

    % NSA ratio
    Nav_ratio = scan.Nav_ratio;

    % == b=0 signal
    b0signal = 1*(exp(-TE/opts.T2));


    % == b\=0 signal
    
    switch modeltype

        case 'Original VERDICT'

            % Tissue parameter vector
            tps = [tissue_params.fRs, tissue_params.fEES, tissue_params.fVASC];
            
            % Diffusion weighting fraction
            fd = simulateSignal( ...
                tps, ...
                scan_params, ...
                modeltype,...
                Rs = tissue_params.Rs...
                );

            % b\=0 signal
            bsignal = b0signal*fd;



        case 'No VASC VERDICT'

            % Tissue parameter vector
            tps = [tissue_params.fRs, tissue_params.fEES];
            
            % Diffusion weighting fraction
            fd = simulateSignal( ...
                tps, ...
                scan_params, ...
                modeltype,...
                Rs = tissue_params.Rs...
                );

            % b\=0 signal
            bsignal = b0signal*fd;


        case 'RDI'

            % Tissue parameter vector
            tps = [tissue_params.fRdists, tissue_params.fEES];
            
            % Diffusion weighting fraction
            fd = simulateSignal( ...
                tps, ...
                scan_params, ...
                modeltype,...
                Rs = tissue_params.Rs,...
                muRs = tissue_params.muRs,...
                sigmaRs = tissue_params.sigmaRs...
                );

            % b\=0 signal
            bsignal = b0signal*fd;


        case 'RDI v1.3'

            % Tissue parameter vector
            tps = [tissue_params.fIC,tissue_params.muR, tissue_params.fEES];
            
            % Diffusion weighting fraction
            fd = simulateSignal( ...
                tps, ...
                scan_params, ...
                modeltype,...
                Rs = tissue_params.Rs,...
                muRs = tissue_params.muRs,...
                sigmaRs = tissue_params.sigmaRs...
                );

            % b\=0 signal
            bsignal = b0signal*fd;



        case 'RDI v1.4'

            % Tissue parameter vector
            tps = [tissue_params.fIC,tissue_params.muR, tissue_params.sigmaR, tissue_params.fEES];
            
            % Diffusion weighting fraction
            fd = simulateSignal( ...
                tps, ...
                scan_params, ...
                modeltype,...
                Rs = tissue_params.Rs,...
                muRs = tissue_params.muRs,...
                sigmaRs = tissue_params.sigmaRs...
                );

            % b\=0 signal
            bsignal = b0signal*fd;


    end

    
    switch opts.noisetype

        case 'Ratio'
            % Generate signal distribution
            [signalDist, zs] = RatioDistRician(b0signal, bsignal, opts.sigma0, Nav_ratio = Nav_ratio);

        case 'Rice'
            % Generate signal distribution
            [signalDist, zs] = RiceDist(b0signal, bsignal, opts.sigma0, Nav_ratio = Nav_ratio);
    
    end
    % Sample signal distribution
    sample = sampleDistribution(signalDist, zs);
    
    % Append to signal array
    signals(scanIndx) = sample;

end


end


