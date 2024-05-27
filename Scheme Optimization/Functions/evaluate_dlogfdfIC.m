function experiment = evaluate_dlogfdfIC(experiment)

arguments
    % Structure containing experiment detail
    experiment
end

scheme = experiment.scheme;
priorgrid = experiment.priorgrid;


% Loop over scans/scheme elements
for scanIndx = 1:length(scheme)

    % Read scan params
    scan_params = [
        scheme(scanIndx).delta,...
        scheme(scanIndx).DELTA,...
        scheme(scanIndx).bval,...
        scheme(scanIndx).TE,...
        scheme(scanIndx).TR,...
        scheme(scanIndx).Nav_ratio
        ];
    % Loop over prior grid of tissue parameters
    for paramsIndx = 1:size(priorgrid,1)

        fIC = priorgrid(paramsIndx, 1);
        R = priorgrid(paramsIndx, 2);
        fEES = 1 - fIC;

        tissue_params = [fIC, R, fEES];

        % define parameter step size
        dfIC = experiment.options.fICstepsize;
        
        % Define step up and down params (case dependent)
        paramsUp = tissue_params;
        paramsDown = tissue_params;
        
        if tissue_params(1) == 0
        
            paramsUp(1) = paramsUp(1) + dfIC;
            paramsUp(3) = paramsUp(3) - dfIC;
            deltaparam = dfIC;
        
        elseif tissue_params(2) == 1
        
            paramsDown(1) = paramsDown(1) - dfIC;
            paramsDown(3) = paramsDown(3) + dfIC;
            deltaparam = dfIC;
        
        else
            paramsUp(1) = paramsUp(1) + dfIC;
            paramsDown(1) = paramsDown(1) - dfIC;
            paramsUp(3) = paramsUp(3) - dfIC;
            paramsDown(3) = paramsDown(3) + dfIC;
            deltaparam = 2*dfIC;
        
        end


        signalDistUp = signalDistribution( ...
        scan_params, ...
        paramsUp, ...
        modeltype = experiment.modeltype,...
        T2 = experiment.T2,...
        sigma = experiment.sigma0) + eps;
        
        signalDistDown = signalDistribution( ...
            scan_params, ...
            paramsDown, ...
            modeltype = experiment.modeltype,...
            T2 = experiment.T2,...
            sigma = experiment.sigma0) + eps;

        % == Evaluate logarithm of signal distributions

        logsignalDistUp = log(signalDistUp);
        logsignalDistDown = log(signalDistDown);
        
        % == Evaluate first order partial derivative
        % u' = (1/2h)*(u(x+h)-u(x-h))
        logsignalDist_partial1 = (1/deltaparam)*(logsignalDistUp - logsignalDistDown);

        if and(scanIndx == 1, paramsIndx == 1)
            experiment.dlogfdfICs = zeros( length(scheme), length(priorgrid), length(signalDistUp));
        end

        experiment.dlogfdfICs(scanIndx, paramsIndx, :) = logsignalDist_partial1;


    end


end



end