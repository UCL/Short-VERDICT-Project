function precon
%PRECON Philips recon  
%   Detailed explanation goes here
%
% Assume 2D, Cartesian, EPI
%
% Remaining issues - SENSE and image quality.
% Why is there an extra apparently an extra pixel shift beyond location
% spectrum offset?
%
%
% Look at recon of SENSE scan - this wrap of YRange may be wrong for EPI.

addDC = true ;

disp(['Select lab for main scan, then SENSE ref, then coil survey'])
fn = pref_uigetfile('UNDISTORT', 'precon') ;
if ~exist(fn,'file')
    return
else
    [fpp, fpfn, fpext] = fileparts(fn) ;
    disp(['File: ',fpfn])
end

MR = MRecon(fn) ;

if MR.Parameter.Scan.SENSEFactor(2) > 1
    isSENSE = true ;
    disp(['Select SENSE ref and then Coil Survey scans'])
    MRref = MRecon(pref_uigetfile('UNDISTORT','precon')) ;
    MRcoilsvy = MRecon(pref_uigetfile('UNDISTORT','precon')) ;
else
    disp(['No SENSE in PE1 detected'])
    isSENSE = false ;
end


MR.Parameter.Scan % Scan parameters

MR.Parameter.Encoding
KyRange = MR.Parameter.Encoding.KyRange ;
YRange = MR.Parameter.Encoding.YRange ;
KxRange = MR.Parameter.Encoding.KxRange ;
XRange = MR.Parameter.Encoding.XRange ;

MR.Parameter.Labels
MR.Parameter.Parameter2Read

MR.Parameter.Parameter2Read.typ = 1 ; % standard data
MR.Parameter.Parameter2Read.Update;  % copied from manual

nloca = length(MR.Parameter.Parameter2Read.loca) ;
islice = ceil(nloca/2) 

MR.Parameter.Parameter2Read.loca = islice -1 ; % 0-based
MR.Parameter.Parameter2Read.extr1 = 0 ;
MR.Parameter.Parameter2Read.extr2 = 0 ;
MR.Parameter.Parameter2Read.aver = 0 ;

MR.ReadData ;
disp(['After MR.ReadData, size(MR.Data): ',num2str(size(MR.Data))])
disp(['x y z coils dyn cardiac phases ech loca mix extr1 extr2 av'])


disp(['Do random phase, PDA, DC and meas phase corrections, (before sort)'])
MR.RandomPhaseCorrection;
MR.PDACorrection;
MR.DcOffsetCorrection;
MR.MeasPhaseCorrection;


MR.SortData ; % converts from nx x nlines_total in time order, to 12D matrix
disp(['After MR.SortData, size(MR.Data): ',num2str(size(MR.Data)),' curFOV: ',num2str(MR.Parameter.Scan.curFOV)])
pshow(MR,1)

disp(['x y z coils dyn cardiac phases ech loca mix extr1 extr2 av'])
disp(['Encoding KyRange: [',num2str(KyRange(1)),' ',num2str(KyRange(2)),']'])
nacq = KyRange(2)-KyRange(1)+1 ;
nfill = 2*KyRange(2) + 1;

% Infer partial Fourier
disp(['Expected partial Fourier: nkyacq / nfilled = ', ...
    num2str(nacq), ' / ',num2str(nfill),' = ',num2str(nacq/nfill)])

% Infer location_spectrum_offsets
nrecony = YRange(2)-YRange(1)+1 ;
nreconx = XRange(2)-XRange(1)+1 ;

if mod(nrecony,2) ~= 0 ||  mod(nreconx,2) ~= 0
    warning(['Odd recon length. nrecony: ',num2str(nrecony),'  nreconx: ',num2str(nreconx)])
end

lo1y = -nrecony/2 -     YRange(1) ;
lo2y =  nrecony/2 - 1 - YRange(2) ;

lo1x = -nreconx/2 - XRange(1) ;
lo2x = nreconx/2 - 1 - XRange(2) ;

if lo1x ~= lo2x || lo1y~=lo2y
    warning(['Inferrred location spectrum offsets not consistent.'])
else
    disp(['location_spectrum_offsets  x: ',num2str(lo1x),'  y: ',num2str(lo1y)])
end

% sin file correspondence
% Oversampling is oversample_factors / sense_extra_ovs_factors
% 'normal' sense factor is sense_factors / sense_extra_ovs_factors
%
% From min_encoding_numbers and max_encoding_numbers fill in partial
% Fourier (e.g. in EPI [-9 27] goes to [-27 27] i.e. odd number PEs
% The 55 is zerofilled to 136. The 136 is calculted:
% recon_resolutions (or output_resolutions?) * Oversampling  e.g. 224 to
% 270.
% Then divide by the 'normal' SENSE factor e.g. to 136
% YRange is then [-136/2 136/2-1] plus location_spectrum_offset
% The 136 is then used to ZeroFill the k-space.
%



% Coil combine and display
eshow(sum(abs(MR.Data),4),'Name','K-space Sum over coils')


MR.GridData ; % Re-grids for non-uniform FE sampling (needs later image corrn)
disp(['After GridData, size(MR.Data): ',num2str(size(MR.Data)),' curFOV: ',num2str(MR.Parameter.Scan.curFOV)])


MR.RingingFilter;
disp(['After RingingFilter, size(MR.Data): ',num2str(size(MR.Data))])

MR.Parameter.Recon.EPICorrectionMethod='nonlin';
MR.Parameter.Recon.EPI2DCorr='yes';

if addDC
    disp(['Adding k-space offset.'])
    kk = MR.Data ;
    kmean = mean(abs(kk(:))) ;
    
    MR.Data = MR.Data + kmean ; %* ones(size(MR.Data),'like',MR.Data)
end

% MR.ZeroFill ;
% disp(['After ksp ZeroFill, size(MR.Data): ',num2str(size(MR.Data)),' curFOV: ',num2str(MR.Parameter.Scan.curFOV)])

pshow(MR,2)

MR.K2IM;
disp(['After K2IM, size(MR.Data): ',num2str(size(MR.Data))])


MR.EPIPhaseCorrection;
disp(['After EPIPhaseCorrection, size(MR.Data): ',num2str(size(MR.Data))])

MR.K2IP;
disp(['After K2IP, size(MR.Data): ',num2str(size(MR.Data))])

pshow(MR,3)

mrecon2dicom(MR) ;
MR.Parameter.ImageInformation


MR.GridderNormalization;
disp(['After GridderNormalization, size(MR.Data): ',num2str(size(MR.Data))])

%%% Doing SENSE first to decrease the computational load later
if isSENSE
    S = MRsense( MRref, MR, MRcoilsvy );
    S.Smooth=1;
    S.Mask=1;
    S.Extrapolate=1;
    
    MR.Parameter.Recon.SENSERegStrength=1e-3;
    %DA S.MatchTargetSize=1;
    S.Perform;
    MR.Parameter.Recon.Sensitivities = S;
    
    MR.SENSEUnfold;
    disp(['After SENSEUnfold, size(MR.Data): ',num2str(size(MR.Data))])
    mrecon2dicom(MR) ;
    pshow(MR, 4)
end

MR.PartialFourier; % Data already zero-filled, this just does homodyne?
disp(['After PartialFourier, size(MR.Data): ',num2str(size(MR.Data)),' curFOV: ',num2str(MR.Parameter.Scan.curFOV)])

MR.RemoveOversampling;
disp(['After RemoveOversampling, size(MR.Data): ',num2str(size(MR.Data))])
disp(['curFOV: ',num2str(MR.Parameter.Scan.curFOV)])
mrecon2dicom(MR) ;

MR.ZeroFill;
disp(['After ZeroFill, size(MR.Data): ',num2str(size(MR.Data))])
disp(['curFOV: ',num2str(MR.Parameter.Scan.curFOV)])
mrecon2dicom(MR) ;

pshow(MR, 5)

eshow(MR.Data)

%
%% PartialFourier always after SENSEunfold to retain phase


% MR_pre = MR.Copy ;
% MR.Compare(MR_pre) 








end

