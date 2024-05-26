disp('Select all single frame DICOM files')

dinfo = datparse ;

[vmfa, mmfa] = d2mat(dinfo,{'slice','fa'},'op','fp') ;

fit = MFAT1(vmfa, mmfa.faVec, mmfa.RepetitionTime_Vec, [], 'calc_nl', false) ;

T1display(fit.T1_lin)

