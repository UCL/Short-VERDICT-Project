
% dinfo and T1 for IR in data.mat
dve = 'I:' ;
fold = 'data\T1MAPPING\TO5_19Dec2014' ;
datfile = 'data.mat' ;

datdir = fullfile(dve,fold) ;

load(fullfile(datdir,datfile)) % loads dinfo volir mir

[vfa, mfa, locfa] = d2mat(dinfo,{'slice','fa','wfio'}, ...
       'series',[1901 2001 2101],'wfio',3,'op','fp') ;

vfar = dreslice(vfa,mfa,mir) ;

% B1 maps 301 - 1101
sb1s = [301:100:1101] ;

nsb1 = length(sb1s) ;

T1all = zeros([size(T1IR,1) size(T1IR,1) nsb1+1]) ;

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

T1all(:,:,nsb1+1) = T1IR ;

%T1display(fit.T1_lin)
