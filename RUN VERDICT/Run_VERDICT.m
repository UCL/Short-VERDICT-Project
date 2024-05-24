% MATLAB script to run VERDICT processing on specified patients

%% Define study path (path to folder containing patients)
% 
% % INNOVATE
% STUDY_path = "C:\Users\adam\OneDrive\Desktop\INNOVATE STUDY COHORT VERDICT IMAGES";

% % Patient volunteers (Full VERDICT)
% STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\Patient Volunteers\Original VERDICT";

% Patient volunteers (Short VERDICT)
STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\Patient Volunteers\Short VERDICT";

%% Define patient numbers

% FOR VOLUNTEERS
PatNums = {"shV_20240410"};

% % FOR INNOVATE
% x = dir(STUDY_path);
% PatNums = {x(:).name};
% PatNums = transpose(PatNums(3:end));

%% DEFINE FOLDERS

% Define output folder
OutputFolder = string(fileread("C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\output_folder.txt"));

% Define schemes folder
schemesfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Schemes";

% Define models folder
modelsfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\MLP Models";

% Define python folder
pythonfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Python";


%% DEFINE VERDICT PROTOCOL

% === Model type
modeltype = 'Original VERDICT'; 

% === Scheme name
schemename = 'Short Scheme v1';

% === fitting technique
fittingtechnique =   'MLP';

% === Noise used in MLP training
noisetype='Rice';
sigma0train = 0.05;
T2train = 10000;

% === ADC
calcADC = true;
% Max b value
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
        calcADC = calcADC,...
        vADCbmax = vADCbmax...
        );
end




