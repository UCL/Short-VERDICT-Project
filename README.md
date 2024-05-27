# ALL CODE WRITTEN FOR SHORT VERDICT PROJECT

Author: Adam Phipps

Email: rmapajp@ucl.ac.uk


===========================================================================

## MATLAB REQUIREMENTS

MATLAB version R2022a or later for compatibility with Python version 3.10

Required Toolboxes:

    - Statistics and Machine Learning Toolbox
    - Image Processing Toolbox
    - Optimization Toolbox

===========================================================================

## PYTHON REQUIREMENTS

Install Python 3.10

Install following packages:

    - numpy
    - scipy
    - matplotlib
    - scikit-learn
    - pydicom
    - openpyxl
    - pandas
    - scikit-image

(run command: 'python -m pip install NAME' in folder where python executable is saved)

Run command: 'pyenv' in MATLAB command line to see if python environment is loaded in MATLAB

MATLAB documentation for Python environment: https://uk.mathworks.com/help/matlab/ref/pyenv.html

If on a Mac and `pyenv` in MATLAB reveals a version that is to old (e.g. in XCode), see 
information about Python versions: https://mac.install.guide/python/
One option to install homebrew, then install the desired Python version:
```
brew install python@3.10
pip3.10 install numpy
...
```
Then the MATLAB command needed will be something like: `pyenv(Version="/usr/local/bin/python3.10")`


===========================================================================



===========================================================================

## RUN VERDICT PROCESSING

To run VERDICT processing, run script: **RUN VERDICT/Run_VERDICT.m**

Before running this script, ensure:
    
    1. STUDY_path is defined. 

This is the parent folder for all patients included in given study. File structure in the folder
should be: STUDY_path/PatNum/...images...

    2. Patient numbers are defined 

(as they appear in STUDY_path folder structure)

    3. Output folder is defined. 

This is where the output images from VERDICT processing will be saved (defined in text document ...\output_folder.txt')

    4. Scheme folder is defined

This is the folder in which schemes are defined as .mat files. 

    5. Python folder is defined.

This is the folder in which python scripts for calling MLP models are saved.

    6. VERDICT protocol is defined

modeltype: 'Original VERDICT', 'No VASC VERDICT', 'RDI v1.3' (which VERDICT model is used in processing)

schemename: 'Original Full', 'Original ex905003000', 'Short Scheme v1' (the acquisition scheme used for processing)

fittingtechnique: 'AMICO', 'MLP' (the method used for model fitting)

(If fittingtechnique set as 'MLP', define 'noisetype', 'T2train', and 'sigma0train' to choose relevant MLP model.)

If ADC calculation also wanted, set calcADC = true and define maximum b value to use in calculation.

===========================================================================

## TRAINING A NEW MLP NETWORK

    1. Create training data for model

Run script **...Short-VERDICT-Project/Model Fitting/MLP/MATLAB/createNewTrainingData.m**

First, ensure that:

    - Ntrain is defined (number of training samples)

    - Parameters for image noise are defined ('noisetype', 'T2', 'sigma0')

    - Training data folder is defined (where data will be saved)

    - Protocols to create traingin data for are defined (modeltypes, schemenames) (cell arrays of char vectors)

(Meta data for creating training data saved as: Training data folder/.../META.mat)


    2. Train MLP network

Run function **trainMLP(modeltype, schemename, opts)**

(...Short-VERDICT-Project/Model Fitting/MLP/MATLAB/trainMLP.m)

Each time function is run, ensure that:

    - modeltype and schemename define correctly

    - Parameters for image noise are define in opts (noisetype, sigma0train, T2train)

    - Following folders are correctly defined:

        pythonfolder - where python scripts are saved
        TrainDataFolder - where training data is saved
        ModelFolder - where MLP model will be saved

    - Parameters for MLP network architecture are specified as desired (Nlayer, Nnode, batchsize)

