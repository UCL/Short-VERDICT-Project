function experiment = evaluate_pdf(experiment)

arguments
    % Structure containing experiment detail
    experiment
end

% Function evaluates pdf for each scan in scheme, for each element of the
% prior grid.
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

        % Evaluate distribution
        [pdf, signals] = signalDistribution( ...
            scan_params, ...
            tissue_params, ...
            modeltype = experiment.modeltype, ...
            T2 = experiment.T2,...
            sigma = experiment.sigma0...
            );


        if and(scanIndx == 1, paramsIndx == 1)
            experiment.pdfs = zeros( length(scheme), length(priorgrid), length(signals));
            experiment.zs = signals;
        end

        experiment.pdfs(scanIndx, paramsIndx, :) = pdf;

    end

end


end