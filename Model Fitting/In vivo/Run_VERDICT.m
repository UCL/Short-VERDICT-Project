% MATLAB script to run VERDICT processing on specified patients

%% Define study path (path to folder containing patients)
% 
% % INNOVATE
% STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\INNOVATE STUDY COHORT VERDICT IMAGES";

% Patient volunteers (Full VERDICT)
STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Imaging Data\Patient Volunteers\Original VERDICT";

% % Patient volunteers (Short VERDICT)
% STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Imaging Data\Patient Volunteers\Short VERDICT";


%% Define patient numbers

PatNums = {"shV_20240307"};% "INN_175"; "INN_145"; "INN_209"; "INN_241"];

% % Read list of patents with downloaded data
% PatTable = readtable("C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\DownloadedPatNums.xlsx", ReadVariableNames=false);
% 
% PatNums = string( PatTable{:,1} );


% x = dir(STUDY_path);
% PatNums = {x(:).name};
% PatNums = transpose(PatNums(3:end));

%% Define output folder

% Define output folder
OutputFolder = string(fileread("C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\output_folder.txt"));

% Define schemes folder
schemesfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Schemes";

% Define models folder
modelsfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\MLP Models";

% Define python folder
pythonfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Python";

%% ======== Define protocol: model type, schemename, fitting technique


% === Model type
modeltype = 'Original VERDICT'; 

% === Scheme name
% schemename =   'Original Full';
schemename = 'Original ex905003000';
% schemename = 'Short Scheme v1';

% fitting technique
fittingtechnique =   'MLP' ;

% Noise used in MLP training
noisetype='Rice';
sigma0train = 0.05;
T2train = 10000;


% T2
calcT2 = false;

% ADC
calcADC = true;
vADCbmax = 1001;


% Run VERDICT processing
for indx = 1:size(PatNums,1)
    PatNum = PatNums{indx};
    disp(["----------->" PatNums(indx)])
    VERDICT( ...
        PatNum, ...
        modeltype = modeltype,...
        schemename = schemename,...
        fittingtechnique = fittingtechnique,...
        noisetype=noisetype,...
        sigma0train = sigma0train,...
        T2train=T2train,...
        STUDY_path=STUDY_path,...
        parent_folder=OutputFolder,...
        schemesfolder = schemesfolder,...
        modelsfolder = modelsfolder,...
        pythonfolder = pythonfolder,...
        calcT2 = calcT2,...
        calcADC = calcADC,...
        vADCbmax = vADCbmax...
        );
end



