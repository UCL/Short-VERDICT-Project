function signal = simulateSignal(tissue_params, scan_params, modeltype, opts)


arguments
    tissue_params % Vector of volume fractions (size dependent on model type!)

    % verdict [fR1, fR2, ... , fRN, fEES, fVASC]
    % noVASC [fR1, fR2, ..., fRN, fEES]
    % RDI [fRdist1, fRdist2, ..., fRdistN, fEES]
    % RDI v1.3 [fIC, R, fEES]
    % RDI v1.4 [fIC, muR, sigmaR, fEES]

    scan_params % Diffusion settings of scan [delta, Delta, b]
    modeltype % specifies model used in fitting

    % == Options

    % verdict fitting
    opts.Rs = linspace(0.1,15.1,17) % Each radii volume fraction is a model parameter
    opts.dIC = 2
    opts.dEES = 2
    opts.dVASC = 8

    % RDI fitting
    opts.muRs = [5, 7.5, 10]; % Radii distributions means
    opts.sigmaRs = [2, 2, 2]; % Radii distribution standard deviations

end

% Unpack scan parameters
delta = scan_params(1);
Delta = scan_params(2);
bval = scan_params(3);
G = stejskal(delta,Delta,bval = bval);

switch modeltype


    case 'Original VERDICT'

         % Radii volume fractions
         fRs = tissue_params(1:end-2);
         
         % Extracellular volume fraction
         fEES = tissue_params(end-1);

         % Vascular volume fraction
         fVASC = tissue_params(end);
         
         % Initialise signal
         signal = 0;

         % Radii/SphereGPD signal
         for radiiIndx = 1:length(fRs)

            fR = fRs(radiiIndx);
            Rsignal = fR*sphereGPD(delta, Delta, G, opts.Rs(radiiIndx), opts.dIC);
            
            signal = signal + Rsignal;
         end

         % Extracellular signal
         EESsignal =  fEES*ball(bval, opts.dEES);
         signal = signal + EESsignal;

         % Vascular signal
         VASCsignal = fVASC*astrosticks(bval, opts.dVASC);
         signal = signal + VASCsignal;



    case 'No VASC VERDICT'

         % Radii volume fractions
         fRs = tissue_params(1:end-1);
         
         % Extracellular volume fraction
         fEES = tissue_params(end);
         
         % Initialise signal
         signal = 0;

         % Radii/SphereGPD signal
         for radiiIndx = 1:length(fRs)

            fR = fRs(radiiIndx);
            Rsignal = fR*sphereGPD(delta, Delta, G, opts.Rs(radiiIndx), opts.dIC);
            
            signal = signal + Rsignal;
         end


         % Extracellular signal
         EESsignal =  fEES*ball(bval, opts.dEES);
         signal = signal + EESsignal;



    case 'RDI'

        % Radii distribution volume fractions
        fRdists = tissue_params(1:end-1);

        % Extracellular volume fraction
        fEES = tissue_params(end);

        signal = 0;

        % Radii distribution signals
        for distIndx = 1:length(opts.muRs)

            % Generate distribution
            fRdist = normpdf(opts.Rs, opts.muRs(distIndx), opts.sigmaRs(distIndx));

            % Normalise and scale
            fRdist = ( fRdists(distIndx)/sum(fRdist) )*fRdist;

            % Loop over radii and append signal
            for Rindx = 1:length(opts.Rs)
                
                R = opts.Rs(Rindx);
                fR = fRdist(Rindx);

                Rsignal = fR*sphereGPD(delta, Delta, G, R, opts.dIC);

                signal = signal + Rsignal;


            end
                
        end

         % Extracellular signal
         EESsignal =  fEES*ball(bval, opts.dEES);
         signal = signal + EESsignal;


    case 'RDI v1.3'

        fIC = tissue_params(1);
        R = tissue_params(2);
        fEES = tissue_params(3);

        % Intracellular signal
        ICsignal = fIC*sphereGPD(delta, Delta, G, R, opts.dIC);

        % Extracellular signal
        EESsignal = fEES*ball(bval, opts.dEES);

        % Total signal
        signal = ICsignal + EESsignal;


    case 'RDI v1.4'

        fIC = tissue_params(1);
        muR = tissue_params(2);
        sigmaR = tissue_params(3);
        fEES = tissue_params(4);

        % == Intracellular signal

        % Generate fRs distribution
        fRs = normpdf(opts.Rs, muR, sigmaR);
        fRs = fIC*(fRs/sum(fRs));
        
        ICsignal = 0;
        for Rindx = 1:length(fRs)
            ICsignal = ICsignal + fRs(Rindx)*sphereGPD(delta, Delta, G, opts.Rs(Rindx), opts.dIC);
        end

        % == Extracellular signal
        EESsignal = fEES*ball(bval, opts.dEES);

        % Total signal
        signal = ICsignal + EESsignal;

end
end