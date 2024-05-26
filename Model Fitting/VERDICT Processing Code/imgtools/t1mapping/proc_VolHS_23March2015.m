


% dinfo = datparse ;
% [vir, mir] = d2mat(dinfo,{'series','itype'},'series', [901:100:2001 ], 'itype',6,'op','fp') ;
% fit.opt = 'lsqnl_mag3' ;
% fit = IRT1(vir, mir.tiVec_indata, fit.opt) ;
% T1display([fit.T1])
% 
% eshow(fit.RB)
% eshow(fit.M0)
% 
% [vb1,matb1,locb1] = d2mat(dinfo,{'slice', 'itype'},'series',2401, 'itype',10,'op','dv') ;
% 
% [vmfa, mmfa, locmfa] = d2mat(dinfo,{'slice','fa','wfio'},'series',[2101 2201 2301],'wfio',3,'op','fp') ;
% vmfar = dreslice(vmfa,mmfa,mir) ;
% vb1r = dreslice(vb1,matb1,mir) ;
% 
% fitmfa = MFAT1(vmfar, mmfa.faVec, mmfa.RepetitionTime_Vec,vb1r,'calc_nl',false) ;
% 
% T1display([fit.T1 fitmfa.T1_lin])
% 
% save data dinfo fit fitmfa locb1 locmfa matb1 mir mmfa vb1 vb1r vir vmfa vmfar



dve = 'I:' ;
fold = 'data\T1MAPPING\VolHS23_Mar2015' ;
datfile = 'data.mat' ;  

datdir = fullfile(dve,fold) ;

load(fullfile(datdir,datfile)) % loads dinfo etc



  
 txtstr = [ '3 flip angles'];
  
 writeDicom(fitmfa.T1_lin, 'T1map', 'header', {dinfo, locmfa(1)},...
      'geom',mir.geom(1), 'FrameOfReferenceUID','keep',...
      'SeriesDescription',['MATLAB T1 from ',txtstr], ...
      'folder_name',datdir ,'fnstem',[txtstr,'_T1'])
  
  
txtstr = [ 'Inv Recov'];
  
 writeDicom(fit.T1, 'T1map', 'header', {dinfo, locmfa(1)},...
      'geom',mir.geom(1), 'FrameOfReferenceUID','keep',...
      'SeriesDescription',['MATLAB T1 from ',txtstr], ...
      'folder_name',datdir ,'fnstem',[txtstr,'_T1'])
  

%T1display(fit.T1_lin)

%%  Volume

s_fa = [1401 1501 1601] ;

dve = 'I:' ;
fold = 'data\T1MAPPING\20150112_hs' ;
datfile = 'dinfo.mat' ;  

datdir = fullfile(dve,fold) ;

load(fullfile(datdir,datfile)) % loads dinfo T1ir vir mir locir ITYPE s_ir

[vfa, mfa, locfa] = d2mat(dinfo,{'slice','fa','wfio'}, ...
       'series',s_fa,'wfio',3,'op','fp') ;
   
 sb1 = 1801;
 [vb1,matb1] = d2mat(dinfo,{'slice', 'itype','series'},'series',sb1, ...
     'itype',10,'op','dv') ;
 
 vb1r = dreslice(vb1,matb1,mfa) ;
 
 fit = MFAT1(vfa, mfa.faVec, mfa.RepetitionTime_Vec, vb1r,'calc_nl',false) ;
 
 T1 = fit.T1_lin ;
 nz = size(T1,3) ;
 
 newUID = dicomuid;
 
 for iz = 1: nz
     txtstr = [ 'sl_',num2str(iz)];
     
     writeDicom(T1(:,:,iz), 'T1map', 'header', {dinfo, locfa(iz)},...
      'geom',mfa.geom(iz), 'FrameOfReferenceUID','keep',...
      'quiet',true, ...
      'SeriesDescription',['MATLAB T1 '], 'SeriesInstanceUID', newUID, ...
      'folder_name',datdir ,'fnstem',[txtstr,'_T1'])
 end
 