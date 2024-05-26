matfn = pref_uigetfile('fionadicom','matfn','*.mat') ; % .mat results filename
multiTEfn = dselect ;  % DICOM file name for multi echo sequence

D = load(matfn,'OUT') ;


LWF = D.OUT(:,:,:,9) ;

warning(['Reversing slice order in LWF!!'])
LWF = LWF(:,:,end:-1:1) ;

dinfo = dmfparse(multiTEfn) ;
[vme, mme, lme] = d2mat(dinfo,{'slice','echo'},'op','fp') ;

writeDicom(LWF, 'LWF','geom', mme.geom, ...
  'FrameOfReferenceUID', 'keep', 'header', {dinfo, lme}, ...
  'ImageComments','Luminal Water Fraction','burn_text',' LWF non-diagnostic ', ...
  'SeriesInstanceUID','new')

