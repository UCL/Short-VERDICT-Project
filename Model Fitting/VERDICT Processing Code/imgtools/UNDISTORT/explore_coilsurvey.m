% Looking at how CoilSurvey data should be handled.
%
%
% Comparison with MRECON'S processing

pn = '/Users/davidatkinson/OneDrive - University College London/grants/CRUKMultiDiscp/CRUKMultiDiscp-shared/software/Demo_code';

rfs = '20170630_115334_B0map_2mmIsoRes_DE' ;
sfs = '20170630_113159_SenseRefScan' ;
cfs = '20170630_112927_CoilSurveyScan' ;

[rdata, rinfo, sdata, sinfo, csdata, csinfo] = ...
    rawselect('file', fullfile(pn, [rfs,'.sin'])) ;

[isdata, sgeom] = recon_process(sdata, sinfo, 'loc', 1, 'MPS_only', false, ...
    'RemoveFEOversampling', true, 'kbeta',6) ;
[icsdata, csgeom] = recon_process(csdata, csinfo, 'loc', 1, 'MPS_only', false, ...
    'RemoveFEOversampling', true, 'kbeta',5) ;

vcs = icsdata(:,:,:,1) - i*icsdata(:,:,:,2) ;
vs = isdata(:,:,:,:) ;

vcsats = dreslice(vcs, csgeom, sgeom) ;

csens = vs ./ vcsats ; % implicit expansion

% mask
avcsats = repmat(abs(vcsats),[1 1 1 size(vs,4)])  ;
loc = find(avcsats < max(avcsats(:))/100 ) ;
csens(loc) = 0 ;

% cs s alignment??




MR = MRecon( fullfile(pn, [rfs,'.lab'])) ;
MS = MRecon( fullfile(pn, [sfs, '.lab'])) ;
MCS = MRecon( fullfile(pn, [cfs, '.lab'])) ;

% MCS.Perform ; 
% Real data, no coils: [64    64    47     1     1     1     1     2 ]
% Includes image domain masking.

% MS.Perform ;
% Real  [112 112 74]. Some fancy image domain masking applied axially -
% geom correction?

S   = MRsense( MS, MR, MCS );

% size(S.CoilSurvey.Data)  reports [64 44 60] i.e. MPS and removal of
% readout oversampling
% Complex, some streaks, no image masking.
% QBC coil 1 appears more like the MRecon and in places has a higher
% magnitude. Use this ? for now

% Outcome: Use QBC coil 1, loca 1 for now.Add a bit more blurring than
% default in recon_process 5?

% Now SENSE scan: filtering and blurring
%  size(S.RefScan.Data)  [112   112    96    32]
%  size(sdata) [ 128    63    48    32     1     2 ]
%   after recon_process MPS_only  [224   112    96    32]
%   MRecon is a bit more blurry, but otherwise similar 
%  eshow(S.RefScan.Data(:,:,:,17))  vs eshow(idata(:,:,:,17))
% 
% Outcome: SENSE scan use a kbeta of 6, loca 1

% Division - masking and 
% 
% S.Perform
%
% S














% Now read my way to check appearance after coil combining 

S.Smooth=1;
S.Mask=1;
S.Extrapolate=1;



