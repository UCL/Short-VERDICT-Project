
% dinfo and T1 for IR in data.mat

% dinfo = datparse ;
% s_ir = [601:100:1301] ;
% ITYPE = 6 ; % magnitude
% [vir, mir, locir] = d2mat(dinfo,{'InversionTime','itype'},'series',s_ir,'itype',ITYPE,'op','fp') ;
% T1ir = IRT1(vir,mir.tiVec,ITYPE) ;
% save dinfo dinfo T1ir vir mir locir ITYPE s_ir

s_fa = [1401 1501 1601] ;

dve = 'I:' ;
fold = 'data\T1MAPPING\20150112_hs' ;
datfile = 'dinfo.mat' ;  

datdir = fullfile(dve,fold) ;

load(fullfile(datdir,datfile)) % loads dinfo T1ir vir mir locir ITYPE s_ir

[vfa, mfa, locfa] = d2mat(dinfo,{'slice','fa','wfio'}, ...
       'series',s_fa,'wfio',3,'op','fp') ;

vfar = dreslice(vfa,mfa,mir) ;

% B1 maps 1701 - 2201
sb1s = [1701:100:2201] ;

nsb1 = length(sb1s) ;

T1all = zeros([size(T1ir,1) size(T1ir,1) nsb1+1]) ;

for isb1 = 1:nsb1
  sb1 = sb1s(isb1) ;
  [vb1,matb1] = d2mat(dinfo,{'slice', 'itype','series'},'series',sb1, ...
                        'itype',10,'op','dv') ;
                    
  vb1r = dreslice(vb1,matb1,mir) ;

  fit = MFAT1(vfar, mfa.faVec, mfa.RepetitionTime_Vec, vb1r,'calc_nl',false) ;
  
  T1all(:,:,isb1) = fit.T1_lin ;
  
  txtstr = [ 'B1_',num2str(sb1)];
  
  writeDicom(fit.T1_lin, 'T1map', 'header', {dinfo, locfa(1)},...
      'geom',mir.geom(1), 'FrameOfReferenceUID','keep',...
      'SeriesDescription',['MATLAB T1 from ',txtstr], ...
      'folder_name',datdir ,'fnstem',[txtstr,'_T1'])

end

T1all(:,:,nsb1+1) = T1ir ;

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
 