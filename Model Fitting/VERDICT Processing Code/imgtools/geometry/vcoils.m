function vcoils
% VCOILS A look at Virtual Coils 
%
% D.Atkinson@ucl.ac.uk  David Atkinson Univesity College London
%

% pn =  '/Users/davidatkinson/OneDrive - University College London/data/20200922_phantoms/20200922_resphant/pngo/2020_09_22/da_347732' ;
% if ~exist(pn,'dir')
%     pn = uigetdir 
% end
% 
% % 54 has R>1
% scan1 = fullfile(pn,'da_22092020_1732530_54_2_wip_dwi_a_0V4.lab') ;

% % data_path = '/Users/davidatkinson/OneDrive - University College London/data/UNDISTORT/Patient21/documents_20200828' ;
% % 
% % MR=MRecon(fullfile(data_path, 'sp_28032018_1149496_12_2_wip_dwi_0_150_500_1000_orig_nsa6_senseV4.lab'));

ffn = '/Users/davidatkinson/OneDrive - University College London/data/20200922_phantoms/20200922_resphant/pngo/2020_09_22/da_347732/da_22092020_1732530_54_2_wip_dwi_a_0V4.lab' ;

MR = MRecon(ffn) ;

% Restrict to a certian slice (loca)
loca = 5 ;
extr1 = 1 ; 
extr2 = 1 ;
aver = 1; 

% MR = MRecon(scan1) ;

MR.Parameter.Parameter2Read.typ = 1;
MR.Parameter.Parameter2Read.loca = loca-1 ;
MR.Parameter.Parameter2Read.extr1 = extr1-1 ;
MR.Parameter.Parameter2Read.extr2 = extr2-1 ;
MR.Parameter.Parameter2Read.aver = aver-1 ;
MR.Parameter.Parameter2Read.Update;


% Do a normal recon but without coil combine
%MR.Parameter.Recon.ArrayCompression = 'Yes' ;
MR.ReadData;  reconreveal(MR,'output','SE','comment','ReadData')

% The following are not performed explicitly if using ArrayCompression -
% already done
MR.RandomPhaseCorrection;
%MR.RemoveOversampling;
MR.PDACorrection;
MR.DcOffsetCorrection;
MR.MeasPhaseCorrection; reconreveal(MR,'output','SE','comment','MeasPhaseCorrection')

MR.SortData;   reconreveal(MR,'output','SE','comment','SortData')

MR.GridData;  reconreveal(MR,'output','SE','comment','GridData')
MR.RingingFilter; reconreveal(MR,'output','SE','comment','RingingFilter')

MR.ZeroFill;   reconreveal(MR,'output','SE','comment','ZeroFill')

% MR.Data = (0+1e-5i) + ones(size(MR.Data),'like',MR.Data) ;

MR.K2IM;       reconreveal(MR,'output','SE','comment','K2IM')
MR.EPIPhaseCorrection;
MR.K2IP;       reconreveal(MR,'output','SE','comment','K2IP')

MR.GridderNormalization;
MR.SENSEUnfold;     reconreveal(MR,'output','SE','comment','SENSEUnfold')
MR.PartialFourier;  reconreveal(MR,'output','SE','comment','PartialFourier')


eshow(MR.Data(:,:,1,:,1,1,1,1,1,1),'Name','after PartialFourier')

return

srf = 'da_22092020_1731313_1000_19_wip_senserefscanV4.lab' ;
crf = 'da_22092020_1730313_1000_14_wip_coilsurveyscanV4.lab' ;

ref = MRecon(fullfile(pn,srf)) ;
coilsurvey = MRecon(fullfile(pn,crf)) ;


S = MRsense(ref, MR, coilsurvey) 

