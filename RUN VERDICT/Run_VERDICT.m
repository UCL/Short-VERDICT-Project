% MATLAB script to run VERDICT processing on specified patients
% Adapted by David Atkinson from code written by Adam Phipps

dataFolder = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/AdamshVERDICT' ;

% This should be changed by putting within a MATLAB project and then files
% are on path.
codeFolder = '/Users/davidatkinson/matlab/Short-VERDICT-Project' ;

%% Define study path (path to folder containing patients)
% 
% % INNOVATE
% STUDY_path = "C:\Users\adam\OneDrive\Desktop\INNOVATE STUDY COHORT VERDICT IMAGES";

% Patient volunteers (Full VERDICT)
STUDY_path = fullfile(dataFolder, 'Imaging Data','Patient Volunteers','Original VERDICT');

% % Patient volunteers (Short VERDICT)
% STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\Patient Volunteers\Short VERDICT";

%% Define patient numbers

% FOR VOLUNTEERS
% PatNums = {"BAR_003"};%"HMU_066";"HMU_069"; "HMU_076"; "HMU_084"; "HMU_121"; "HMU_176"; "HMU_180"; "HMU_201"};
PatNums = {"shV_20240410"};

% % FOR INNOVATE
% x = dir(STUDY_path);
% PatNums = {x(:).name};
% PatNums = transpose(PatNums(3:end));

%% DEFINE FOLDERS

% Define output folder
relativeOutputFolder = string(fileread(fullfile(codeFolder,'relative_output_folder.txt')));
OutputFolder = fullfile(dataFolder,relativeOutputFolder) ;

% Define schemes folder
schemesfolder = fullfile(codeFolder, 'Model Fitting','Schemes');

% Define models folder
modelsfolder = fullfile(codeFolder,'Model Fitting','MLP','MLP Models');

% Define python folder
pythonfolder = fullfile(codeFolder,'Model Fitting','MLP','Python');


%% DEFINE VERDICT PROTOCOL

% === Model type
modeltype = 'Original VERDICT'; 

% === Scheme name
schemename = 'Original ex905003000';

% === fitting technique
fittingtechnique =   'MLP';

% === Noise used in MLP training
noisetype='Rice';
sigma0train = 0.05;
T2train = 10000;

% === ADC
calcADC = true;
% Max b value
vADCbmax = 1501;


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
        calcADC = calcADC,...
        vADCbmax = vADCbmax...
        );
end




