% Testing sensitivity analysis function

% Read list of patient numbers included in INNOVATE trial
PatNums = dir("C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\INNOVATE\INNOVATE STUDY COHORT VERDICT IMAGES");
PatNums = {PatNums.name};
PatNums = PatNums(3:end);

% Exclude
exclude = {'INN_377', 'INN_378'};

for pat = exclude
    where = (string(PatNums) == pat);
    PatNums(where)=[];
end


% Model types
modeltypes = {'Original VERDICT', 'No VASC VERDICT'};%, 'Original VERDICT', 'Original VERDICT', 'No VASC VERDICT'};

% Scheme names
schemenames =  {'Original Full', 'Original ex905003000'};%, 'Original Full', 'Original ex905003000', 'Original ex905003000'};

% Fitting techniques 
fittingtechniques =  {'AMICO', 'MLP'};%, 'MLP', 'MLP'};

% Parameters
parameters = {'fIC'};

scoretype = "median";

%% Run sensitivity analysis
scores = SensitivityAnalysis(PatNums, modeltypes, schemenames, fittingtechniques, parameters = parameters, scoretype = scoretype);

figure;
plot(1-scores(1).specificity, scores(1).sensitivity, LineWidth = 2)
hold on
plot(1-scores(2).specificity, scores(2).sensitivity, LineWidth = 2)
hold on
% plot(1-scores(3).specificity, scores(3).sensitivity)
% hold on
% plot(1-scores(4).specificity, scores(4).sensitivity)
ylabel('Sensitivity')
xlabel('1-Specificity') 
legend( [ string( ['Original Protocol, AUC: ' num2str(scores(1).AUC(1))]) , ...
    string( ['Reduced Protocol, AUC: ' num2str(scores(2).AUC(1))]), ...
    % string( ['Protocol 3, AUC: ' num2str(scores(3).AUC(1))]), ...
    % string( ['Protocol 4, AUC: ' num2str(scores(4).AUC(1))]) ...
    ])
% legend( [ string( ['Protocol 1, AUC: ' num2str(scores(1).AUC(1))]) , ...
%     string( ['Protocol 2, AUC: ' num2str(scores(2).AUC(1))]), ...
%     string( ['Protocol 3, AUC: ' num2str(scores(3).AUC(1))]), ...
%     string( ['Protocol 4, AUC: ' num2str(scores(4).AUC(1))]) ...
%     ])
% title(['ROC AUC: ' schemenames{1}])

disp('')

%% Threshold and specificity at 90% sensitivity
