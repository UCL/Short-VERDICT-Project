
% pd = '/Volumes/Ashur Pro2/data/20200309_TO5_MR2/TO5_AXIAL_T1/UCH_PROSTATE_20200309_132736_985000' ;
pd = '/Users/davidatkinson/OneDrive - University College London/data/20200309_TO5_MR2/TO5_AXIAL_T1/UCH_PROSTATE_20200309_132736_985000' ;

fn{1} = 'T1_VIBE_11DEG_5MES_ROT_FASTWATER' ;
dn{1} = '11deg fast water';
col{1} = [1 0 0] ;

fn{2} = 'T1_VIBE_11DEG_5MES_ROT_NORMALWATER' ;
dn{2} = '11deg normal water' ;
col{2} = [0 1 0] ;

fn{3} = 'T1_VIBE_15DEG_5MES_ROT' ;
dn{3} = 'std protocol 15deg';
col{3} = [0 0 0];

fn{4} = 'T1_VIBE_11DEG_5MES_ROT' ;
dn{4} = 'std protocol 11deg';
col{4} = [0 0 1];

fn{5} = 'T1_VIBE_11DEG_5MES_ROT_SPAIR' ;
dn{5} = '11deg SPAIR';
col{5} = [1 1 0] ;

fn{6} = 'T1_VIBE_15DEG_5MES_ROT_SPAIR' ;
dn{6} = '15deg SPAIR';
col{6} = [1 0 1] ;

fn{7} = 'T1_VIBE_15DEG_5MES_ROT_NOFATSAT' ;
dn{7} = '15deg nofatsat';
col{7} = [1 0.5 1] ;


fndisp = [1 2 3 4 5 6 7] ;

pix2 = {229, 221} ; % Y, X
pix1 = {233, 157} ; 
fluidp = {'Center',[186.35  197.25],'Radius', 15.17} ;
gel2p = {'Center',[221 229],'Radius', 11} ;
gel1p = {'Center',[157 233],'Radius', 11} ; 


sl = 8 ;
dyn = 5 ;

hf = figure('Name','explore');
hax = gca ;

dyn_noise = 0 ;
gap = 2 ;

 for ifn = 1: length(fndisp)
     dinfo = dparse(fullfile(pd, fn{ifn} )) ;
     [vd, mm] = d2mat(dinfo,{'slice','aqno'},'op','dv') ;
   
     nt = size(vd,4) ;
     
     [mn1, std1]  = circleinf(vd,sl, gel1p) ;
     [mn2, std2]  = circleinf(vd,sl, gel2p) ;
     [mnf, stdf]  = circleinf(vd,sl, fluidp) ;
     
     t1 = squeeze(vd(pix1{1}, pix1{2},sl,1:end-dyn_noise)) ; % exclude last dynamic if noise scan
     t2 = squeeze(vd(pix2{1}, pix2{2},sl,1:end-dyn_noise)) ;
     xt=[1:nt nt+1+gap:(2*nt)+gap] ;
     plot(hax, xt, cat(1,t1,t2),'LineWidth',2,'Color',col{fndisp(ifn)}, 'DisplayName',dn{fndisp(ifn)})
     hold(hax,'on') ;
     plot(hax, xt, cat(1,mn1+std1,mn2+std2),'LineWidth',1,'Color',col{fndisp(ifn)}, 'DisplayName',['mn+ ',dn{fndisp(ifn)}])
     plot(hax, xt, cat(1,mn1-std1,mn2-std2),'LineWidth',1,'Color',col{fndisp(ifn)}, 'DisplayName',['mn- ',dn{fndisp(ifn)}])
            
     axis(hax, [1 (2*nt)+gap 0 300])  
     grid(hax,'on') 
     
 end
 
 legend(hax)
 
 hax.XTick = [1 nt nt+1+gap (2*nt)+gap];
 nts = num2str(nt) ;
 hax.XTickLabel = {'1', nts, '1', nts} ;
        
   
 function [mn, stddev] = circleinf(vd, sl, circlep)
 
 him = figure('Name','circle inf') ;
 img = vd(:,:,sl,1) ;
 him = imshow(img,[]);
 
 % cente region circle
 hroi = drawcircle(circlep{:}) ;   
 bw  = createMask(hroi) ;
 loc = find(bw) ;
 for idyn = 1:size(vd,4)
     img = vd(:,:,sl,idyn) ;
     vals = img(loc) ;
     mn(idyn,1) = mean(vals);
     stddev(idyn,1) = std(vals) ;
 end
 end
     