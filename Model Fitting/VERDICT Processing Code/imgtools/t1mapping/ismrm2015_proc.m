function ismrm2015_proc

dve = 'I:' ;
root_folder = 'data\T1MAPPING\ISMRM2015' ;


currd = pwd ;

%-------------------------------------------------

study_folder = 'EQJC29_Sept2014' ;

s_mfa = [801 901]; dix_type = 'wfio'; dix_num = 3 ;
slc = 64 ;

s_b1 = [401 ] ; B1medfilt = [20 20] ; % 1001 also available
s_diff = [701 ] ;

proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1, B1medfilt, s_diff)
%--------------------------------------------------
study_folder = 'EQ JC28_Oct_14' ;
s_mfa = [701 801]; dix_type = 'wfio'; dix_num = 3 ;
slc = 121-66 ;

s_b1 = [901 ] ; B1medfilt = [20 20] ;
s_diff = [601 ] ;

proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1,   B1medfilt, s_diff)
%       
% %------------------------------------------------
study_folder = 'EQJC04_Nov14' ;
s_mfa = [601 701]; dix_type = 'echo'; dix_num = 1 ;
slc = 60 ;

s_b1 = [501 ] ; B1medfilt = [20 20] ;
s_diff = [801 ] ;

proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1, B1medfilt, s_diff)
% 
% %------------------------------------------------
% %-------------------------------------------------
% 
study_folder = 'EQMSL20_Aug2014' ;
 
s_mfa = [303 403]; dix_type = 'wfio'; dix_num = 3 ;
slc = 71 ;

s_b1 = [503 ] ; B1medfilt = [20 20] ;
s_diff = [1104 ] ;

proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1, B1medfilt, s_diff)
%  
% %-------------------------------------------------
% 
study_folder = 'EQMSL11_Sept2014' ;

s_mfa = [303 403]; dix_type = 'wfio'; dix_num = 3 ;
slc = 70 ;

s_b1 = [503 ] ; B1medfilt = [20 20] ;
s_diff = [1103 ] ;

proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1, B1medfilt, s_diff)
 
% 
% 
% 
% %--------------------------------------------------
study_folder = 'EQMSL30_Oct2014' ;
 
s_mfa = [303 403]; dix_type = 'wfio'; dix_num = 3 ;
slc = 74 ;

s_b1 = [503 ] ; B1medfilt = [20 20] ;
s_diff = [1203 ] ;

proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1, B1medfilt, s_diff)
 
%       
% %------------------------------------------------
% %------------------------------------------------
% study_folder = 'EQHL05_Sept2014' ;
% s_mfa = [501 601]; dix_type = 'wfio'; dix_num = 3 ;
% slc = 121-70 ;
% 
% s_b1 = [401 ] ; B1medfilt = [20 20] ; % 
% s_diff = [701 ] ;
% 
% proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
%           slc, s_b1, B1medfilt, s_diff)

%------------------------------------------------
% study_folder = 'EQHL03_Oct2014' ;
% s_mfa = [601 701]; dix_type = 'wfio'; dix_num = 3 ;
% slc = 121-56 ;
% 
% s_b1 = [501 ] ; B1medfilt = [20 20] ; % 1701also available
% s_diff = [801 ] ;
% 
% proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
%           slc, s_b1, B1medfilt, s_diff)

%-------------------------------------------------
% study_folder = 'EQHL22_Oct2014' ;
% 
% s_mfa = [501 601]; dix_type = 'wfio'; dix_num = 3 ;
% slc = 121-71 ;
% 
% s_b1 = [401 ] ; B1medfilt = [20 20] ; % 1301 also available
% s_diff = [1401 ] ;
% 
% proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
%           slc, s_b1, B1medfilt, s_diff)
 
%----------------------------------------------------------
% study_folder = '20141020_breastmets' ;
% root_folder = 'data\T1MAPPING' ;
% 
% s_mfa = [304 404]; dix_type = 'wfio'; dix_num = 3 ;
% slc = 1 ; % IR slice
% 
% s_b1 = [504 ] ; B1medfilt = [20 20] ; % 1301 also available
% s_diff = [1202 ] ;
% s_ir = [1501:100:2201] ;
% 
% proc_b(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
%           slc, s_b1, B1medfilt, s_diff, s_ir)



end

function proc(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1,   B1medfilt, s_diff)
      
dinfo = load_dinfo(dve, root_folder, study_folder) ;
dinfo = cast_private(dinfo) ;

disp(['!! b1 scaling (should be dv)'])
[vb1,mb1] = d2mat(dinfo,{'slice','itype'},'series',s_b1,'itype',10 ,'op','pv') ;
vb1=single(vb1)/4095*300;

[vmfa, mmfa, locs] = d2mat(dinfo,{'slice','fa',dix_type}, ...
         'series',s_mfa, dix_type, dix_num,'op','fp') ;
[vdiff, mdiff, locs_diff] = d2mat(dinfo,{'slice','bv'},'series',s_diff, 'op','fp') ;

vb1r = dreslice(vb1,mb1,mmfa.geom(slc));
vdiffr = dreslice(vdiff, mdiff, mmfa.geom(slc)) ;

eshow(medfilt2(vb1r,B1medfilt),'medfilt') ;

disp(['NO B! MAP APPLIED'])
% fit_mfa = MFAT1(vmfa(:,:,slc,:),mmfa.faVec, mmfa.RepetitionTime_Vec, ...
%     vb1r,'calc_nl' ,false, 'B1medfilt',B1medfilt ) ;
fit_mfa = MFAT1(vmfa(:,:,slc,:),mmfa.faVec, mmfa.RepetitionTime_Vec, ...
    [],'calc_nl' ,false) ;


T1display(fit_mfa.T1_lin) ;

[ADC, S0] = calcADC(vdiffr, mdiff.bvVec) ;
jloc = find(abs(imag(ADC))>0) ;
ADC(jloc) = 0 ;
ADC(ADC<0) = 0 ;
ADC = abs(ADC) ; % residual imag (but ==0)

displayADC(ADC*1e6, 2000)

txtstr = [ study_folder,'_',num2str(slc)];
writeDicom(fit_mfa.T1_lin, 'T1map', 'header', {dinfo, locs(slc,1)},'geom',mmfa.geom(slc), 'FrameOfReferenceUID','keep','fnstem',[txtstr,'_T1noB1'])
writeDicom(ADC, 'ADCbody', 'header', {dinfo, locs_diff(1)},'geom',mmfa.geom(slc), 'FrameOfReferenceUID','keep','fnstem',[txtstr,'_ADC'])

end

function proc_b(dve, root_folder, study_folder, s_mfa, dix_type, dix_num, ...
          slc, s_b1,   B1medfilt, s_diff, s_ir)
      
dinfo = load_dinfo(dve, root_folder, study_folder) ;


[vb1,mb1] = d2mat(dinfo,{'slice','itype'},'series',s_b1,'itype',10 ,'op','dv') ;

[vmfa, mmfa, locs] = d2mat(dinfo,{'slice','fa',dix_type}, ...
         'series',s_mfa, dix_type, dix_num,'op','fp') ;
[vdiff, mdiff, locs_diff] = d2mat(dinfo,{'slice','bv'},'series',s_diff, 'op','fp') ;


ITYPE = 6 ;  % 6 for magnitude, 8 for real 
[vir, mir, locir] = d2mat(dinfo,{'InversionTime','itype'},'series',s_ir,'itype',ITYPE,'op','fp') ; 
% T1ir = IRT1(vir,mir.tiVec,ITYPE) ;    
% 
% T1display(T1ir) ;


vb1r = dreslice(vb1,mb1,mir);
vdiffr = dreslice(vdiff, mdiff, mir) ;
vmfar = dreslice(vmfa,mmfa,mir) ;

eshow(medfilt2(vb1r,B1medfilt),'medfilt') ;

fit_mfa = MFAT1(vmfar,mmfa.faVec, mmfa.RepetitionTime_Vec, ...
    vb1r,'calc_nl' ,false, 'B1medfilt',B1medfilt ) ;

T1display(fit_mfa.T1_lin) ;

[ADC, S0] = calcADC(vdiffr, mdiff.bvVec) ;
jloc = find(abs(imag(ADC))>0) ;
ADC(jloc) = 0 ;
ADC(ADC<0) = 0 ;
ADC = abs(ADC) ; % residual imag (but ==0)

displayADC(ADC*1e6, 2000)

txtstr = [ study_folder];
writeDicom(fit_mfa.T1_lin, 'T1map', 'header', {dinfo, locs(slc,1)},'geom',mir.geom(slc), 'FrameOfReferenceUID','keep','fnstem',[txtstr,'_T1'])
%writeDicom(T1ir, 'T1map', 'header', {dinfo, locir(slc,1)},'geom',mir.geom(slc), 'FrameOfReferenceUID','keep','fnstem',[txtstr,'_T1IR'])
%writeDicom(ADC, 'ADCbody', 'header', {dinfo, locs_diff(1)},'geom',mir.geom(slc), 'FrameOfReferenceUID','keep','fnstem',[txtstr,'_ADC'])

end


function dinfo = load_dinfo(dve, root_folder, study_folder)
fold = fullfile(dve,root_folder,study_folder) ;
S = load(fullfile(fold,'dinfo.mat')) ;
dinfo = S.dinfo ;
end


