function epiksp

% pn =  '/Users/davidatkinson/OneDrive - University College London/data/20200922_phantoms/20200922_resphant/datalist' ;
% if ~exist(pn,'dir')
%     pn = uigetdir 
% end
% 
% scan1 = fullfile(pn,'raw_054.data') ;

ffn = pref_uigetfile('epiksp','file') ;

loca = 1 

MR = MRecon(ffn) ;


MR.Parameter.Parameter2Read.typ = 1;
MR.Parameter.Parameter2Read.Update;

AutoUpdateStatus = MR.Parameter.Recon.AutoUpdateInfoPars;
MR.Parameter.Recon.AutoUpdateInfoPars = 'no';

MR.ReadData;  reconreveal(MR,'output','SE','comment','ReadData')

MR.RandomPhaseCorrection;
MR.RemoveOversampling;
MR.PDACorrection;
MR.DcOffsetCorrection;
MR.MeasPhaseCorrection;

MR.SortData;   reconreveal(MR,'output','SE','comment','SortData')
                
MR.GridData;   reconreveal(MR,'output','SE','comment','GridData')
MR.RingingFilter;

MR.ZeroFill;   reconreveal(MR,'output','SE','comment','ZeroFill')


MR.Data = (0+1e-5i) + ones(size(MR.Data),'like',MR.Data) ;

MR.K2IM;       reconreveal(MR,'output','SE','comment','K2IM')

%MR.EPIPhaseCorrection;
MR.K2IP;       reconreveal(MR,'output','SE','comment','K2IP')


eshow(MR.Data(:,:,1,1,1,1,1,1,1,1),'Name','after K2IP')



%MR.GridderNormalization;
% MR.SENSEUnfold;
%MR.PartialFourier;
%MR.ConcomitantFieldCorrection;
% MR.DivideFlowSegments;
MR.CombineCoils;
% MR.Average;
% MR.GeometryCorrection;
MR.RemoveOversampling;
%MR.FlowPhaseCorrection;
% MR.ReconTKE;     reconreveal(MR,'output','S','comment','ReconTKE')
MR.ZeroFill;     reconreveal(MR,'output','S','comment','ZeroFilli')
MR.RotateImage;  
reconreveal(MR,'output','S','comment','RotateImage')
disp(' ')



