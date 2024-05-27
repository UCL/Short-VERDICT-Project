function [scheme, Y, fIC, fEES, fVASC, R, rmse] = verdict_Adam(dfolder, output_folder, opts)

arguments

    dfolder
    output_folder

    %% Options

    % % David
    % opts.outputDicom   = 'true' % output results in DICOM files
    % opts.addPhilipsPrivate = 'true' % adds Private fields to Philips if available (for scanner upload)
    % opts.outputMatFile = 'true'  % output variables in one .mat file
    % opts.outputAInReport='false' % Output AMICO matrix A in report

    % opts.swapinvXNAT   = 'false' % swap in the data from a vXNAT.mat file
    % opts.swapinvMAT     = 'false' % swap in MAT file for Guanda's project



    % opts.quiet           = 'false' % suppress figures and report viewer

    % opts.resultsFolderName = ''    % If empty will default to res-datetime
    % opts.fICReconNumber  = '5'   % DICOM fIC recon number for saved fIC (specified in testing)
    % opts.vBaseSeriesNumber  % DICOM Base Series number for saved fIC (specified in testing), default is from b=90
    % % Final fIC Series Number will be 100*vBaseSeriesNumber + fICReconNumber

    % opts.vADCbmax = '1600' % max b-value used in VERDICT ADC calculation

    % 
    % % Set Release/Version tag here (will be output in report and .mat file)
    % opts.verdictVersion = '1.010 - R20230827' ;

    opts.bvSortDirection = 'descend' % to correspond to XNAT pipeline
    opts.allowedSeriesNumbers = [] % SeriesNumbers in this set can be used
    opts.register      = 'true'  % input data will have be registered
    opts.usedirecdiff  = false % use the 3 directional diffusion images
    opts.addb0ToScheme   = 'true'  % adds b=0 (signal of 1) to every series
    opts.forcedSchemeName = ''     % Force scheme name for debugging
    opts.maskhwmm        = '48'    % mask half-width in mm used for registration
    opts.forceXNATscheme = 'false' % ignores scanner and uses XNAT scheme file    

    %% DATA SAVING ADAM

    opts.PatientID = 'PAT_XXX'
    opts.parent_folder = 'no folder'

    %% ADC ADAM
    opts.calcADC = false % Bool of whether to calculate ADC
    opts.vADCbmax  % 


    %% Fitting technique ADAM
    opts.fittingtechnique = 'MLP'
  
    % Solver for AMICO fitting
    opts.solver = 'lsqnonnegTikhonov'

    % Noise model in MLP training
    opts.noisetype 
    opts.sigma0train   
    opts.T2train  

    opts.modelsfolder 
    opts.pythonfolder 

    %% Schemename ADAM
    opts.schemename 
    opts.schemesfolder 
 

    %% Model type ADAM
    opts.modeltype

    %% Fitting ADAM
    opts.fitting;
    % VERDICT: induvidual radii fitting
    % RDI: fitting for radii distributions

    opts.ncompart = 2            % Number of tissue compartment beyond sphere (1 for EES, 2 for EES and VASC)
    opts.Rs = [] % Specified Rs used in fitting
    opts.mask = [] % Mask for fitting



end



% makeDOMCompilable() % Needed for deployed Report Generator

[scheme, Y, fIC, fEES, fVASC, R, rmse] = verdict_process_Adam(dfolder, output_folder, opts) ;

