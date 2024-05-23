function experiment = evaluate_FImatrix(experiment)

% Function evaluates FI matrix for experiment

% Initialise empty matrix (2x2)
FImatrix = zeros(2,2);

scheme = experiment.scheme;
priorgrid = experiment.priorgrid;
pdfs = experiment.pdfs;
dlogfdfICs = experiment.dlogfdfICs;
dlogfdRs = experiment.dlogfdRs;

% Loop over scans/scheme elements
for scanIndx = 1:length(scheme)

    % Read scan params
    scan_params = [scheme(scanIndx).delta, scheme(scanIndx).DELTA, scheme(scanIndx).bval, scheme(scanIndx).TE];

    % Loop over prior grid of tissue parameters
    for paramsIndx = 1:size(priorgrid,1)

        fIC = priorgrid(paramsIndx, 1);
        R = priorgrid(paramsIndx, 2);
        fEES = 1 - fIC;
        tissue_params = [fIC, R, fEES];

        pdf = squeeze( pdfs(scanIndx, paramsIndx,:));
        dlogfdfIC = squeeze( dlogfdfICs(scanIndx, paramsIndx,:) );
        dlogfdR = squeeze( dlogfdRs(scanIndx, paramsIndx,:) );

        %% Matrix element (1,1) (d/dfIC)^2
        M11 = expectedvalue(dlogfdfIC.*dlogfdfIC, pdf, experiment.zs);

        FImatrix(1,1) = FImatrix(1,1) + M11;


        %% Matrix element (1,2) & (2,1) (d/dfIC)*(d/dR)
        M12 = expectedvalue(dlogfdfIC.*dlogfdR, pdf, experiment.zs);

        FImatrix(1,2) = FImatrix(1,2) + M12;
        FImatrix(2,1) = FImatrix(2,1) + M12;

        %% Matrix element (2,2) & (2,1) (d/dR)^2
        M22 = expectedvalue(dlogfdR.*dlogfdR, pdf, experiment.zs);

        FImatrix(2,2) = FImatrix(2,2) + M22;
    end
end

experiment.FImatrix = FImatrix;

end