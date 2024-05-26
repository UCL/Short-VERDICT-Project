% Check the analyss of Saurabh's data

fmaskPZ = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/INNOVATE/XNAT-INNOVATE/INN-065-PCA_20160725/ROIs/INN_165_DMI_normalPZ.nii' ;
fmaskL1 = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/INNOVATE/XNAT-INNOVATE/INN-065-PCA_20160725/ROIs/INN_165_DMI_lesion1.nii' ;

fdorigb1500 = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/INNOVATE/XNAT-INNOVATE/INN-065-PCA_20160725/scans/1001_b1500_vx1.3/DICOM/1.2.826.0.1.1817913.212.1.1.1.54061495-1001-1-bataju.dcm' ;
fnorigb1500 = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/INNOVATE/XNAT-INNOVATE/INN-065-PCA_20160725/scans/1001_b1500_vx1.3/NIFTI/1.2.826.0.1.1817913.212.1.1.1.54061495-1001-1-bataju.nii.gz' ;

fdb1500reg = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/INNOVATE/XNAT-INNOVATE/INN-065-PCA_20160725/assessors/prostate_xnat_E01859/OsiriX/osirix/1001_b1500_vx1.3_reg.dcm' ;

fnfic = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/INNOVATE/XNAT-INNOVATE/INN-065-PCA_20160725/assessors/prostate_xnat_E01860/RMAPS1/FIT_fIC.nii.gz' ;

% Is reg different from orig?
dinfo_origb1500 = dfparse(fdorigb1500) ;
dinfo_b1500reg  = dfparse(fdb1500reg) ;  

% From exploration:
%  For DICOM *_reg, the data is unit8 (bit depth 8, not 12).
%  It does appear slightly shifted wrt the original (hence registered)
%  The data has been put into a DICOM file without updating the header
%  correctly. The b=0 is in b=1500, [ddty=1, bdirec =1]. Followed by the
%  three diffusion weighted images in [ddty=1, bdirec=2]  [ddty=1, bdirec=3] 
%  [ddty=2] 
% 
[vorigb1500, morigb1500] = d2mat(dinfo_origb1500, {'slice','bv', 'ddty'}, ...
    'bv',0,'ddty',0,'op','pv_single') ;

[vb1500reg, mb1500reg]   = d2mat(dinfo_b1500reg,  {'slice', 'bdirec', 'ddty'}, ...
    'bdirec',1,'ddty',1,'op','pv_single') ;


eshow( vorigb1500/max(vorigb1500(:)) - vb1500reg/max(vb1500reg(:)) )