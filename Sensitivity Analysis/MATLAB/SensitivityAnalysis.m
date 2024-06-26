function scores = SensitivityAnalysis(PatNums, modeltypes, schemenames, fittingtechniques, opts)

arguments
    PatNums % list of patient numbers
    modeltypes % model types to analyse 
    schemenames % schemes to analyse
    fittingtechniques % fitting techniques to analyse

    % == Options

    opts.noisetype = 'Rice'
    opts.sigma0train = 0.05 % If MLP fitting being analysed
    opts.T2train = 10000

    opts.parameters = "fIC" % Parameters for lesion characterisation (string or list of strings)
    opts.scoretype = "median"
    opts.classifier = "threshold"

    opts.ROIdrawer = "NT"
    opts.ROIname = "L1_b3000_NT"
    opts.ROIfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\INNOVATE\ROIs"
    
end


%% Read output folder
output_folder = fileread("C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\output_folder.txt");

%% Call Python script to run first half of analysis
% 
pyrunfile( ...
    'RunSensitivityAnalysis.py', ...
    PatNums = PatNums, ...
    modeltypes = modeltypes,...
    schemenames =  schemenames,...
    fittingtechniques = fittingtechniques,...
    noisetype = opts.noisetype,...
    sigma0train = opts.sigma0train,...
    T2train = opts.T2train,...
    parameters = opts.parameters,...
    scoretype = opts.scoretype,...
    classifier = opts.classifier,...
    datafolder = output_folder,...
    ROIdrawer = opts.ROIdrawer,...
    ROIname = opts.ROIname,...
    ROIfolder = opts.ROIfolder...
    )


%% Read scores and biopsy results

% % Create combination grid
% [x,y,z] = meshgrid( ...
%     string(modeltypes), ...
%     string(schemenames), ...
%     string(fittingtechniques) ...
%     );

% combos = [x;y;z];
% combos = combos(:);
% 
% combos = stack(string(modeltypes), ...
%     string(schemenames), ...
%     string(fittingtechniques))

combos = [string(modeltypes); string(schemenames); string(fittingtechniques)];
combos = combos(:);


% Initialise empty structure for scores over combinations
scores = struct();

% Loop over combinations
for cindx = 1:(length(combos)/3)

    modeltype = combos((cindx-1)*3 + 1);
    schemename = combos((cindx-1)*3 + 2);
    fittingtechnique = combos((cindx-1)*3 + 3);
    % 
    % Load scores
    if strcmp(opts.scoretype, "perceptron")

        params_string = '';
        for indx = 1:length(opts.parameters)
            params_string = [params_string char(opts.parameters(indx)) '_'];
        end


        % switch fittingtechnique
        %     case 'MLP'
        %         these_scores = load([char(output_folder) '/ROI Scores/' char(modeltype) '/' char(schemename) '/' char(fittingtechnique) '/sigma0 = ' num2str(opts.sigma0train) '/' char(opts.ROIdrawer) '/' char(opts.ROIname) '/' char(opts.scoretype) '/' params_string '/scoresDF.mat']).scores;
        %     case 'AMICO'
        %         these_scores = load([char(output_folder) '/ROI Scores/' char(modeltype) '/' char(schemename) '/' char(fittingtechnique) '/' char(opts.ROIdrawer) '/' char(opts.ROIname) '/' char(opts.scoretype) '/' params_string '/scoresDF.mat']).scores;
        % end

    elseif strcmp(opts.scoretype, "median")

        switch fittingtechnique
            case 'MLP'
                these_scores = load([char(output_folder) '/ROI Scores/' char(modeltype) '/' char(schemename) '/' char(fittingtechnique) '/' opts.noisetype '/T2_' num2str(opts.T2train) '/sigma_' num2str(opts.sigma0train) '/' char(opts.ROIdrawer) '/' char(opts.ROIname) '/' char(opts.scoretype) '/scoresDF.mat']).scores;
            case 'AMICO'
                these_scores = load([char(output_folder) '/ROI Scores/' char(modeltype) '/' char(schemename) '/' char(fittingtechnique) '/' char(opts.ROIdrawer) '/' char(opts.ROIname) '/' char(opts.scoretype) '/scoresDF.mat']).scores;
        end
    end
    


    scores(cindx).PatNums = these_scores.Patient_ID;
    scores(cindx).scores = these_scores.Score;
    scores(cindx).modeltype = modeltype;
    scores(cindx).schemename = schemename;
    scores(cindx).fittingtechnique = fittingtechnique;
    scores(cindx).biopsyresults = zeros(length(these_scores.Score),1); 

    % Load biopsy results
    BiopsyResults = load([char(output_folder) '\Biopsy Results\BiopsyResultsDF.mat']).BiopsyResults;

    % Fill in biopsy results for scores structure
    BiopsyPatNums = string(BiopsyResults.Patient_ID);
    for patindx = 1:length(BiopsyPatNums)
        PatNum = char(BiopsyPatNums(patindx));
        PatNum = string(PatNum(1:7));
        where =  ( string( scores(cindx).PatNums ) == PatNum );
        scores(cindx).biopsyresults(where) = BiopsyResults.Biopsy_Result(patindx);
    end



    % Sensitivity analysis
    ylabel = scores(cindx).biopsyresults;
    yscore = scores(cindx).scores;
    [X,Y,T, AUC] = perfcurve(ylabel ,yscore, 1, 'Alpha', 0.05, 'NBoot',1000,'XVals', 'All');

    scores(cindx).sensitivity = Y(:,1);
    scores(cindx).specificity = 1-X;
    scores(cindx).thresholds = T(:,1);
    scores(cindx).AUC = AUC;

end










end





