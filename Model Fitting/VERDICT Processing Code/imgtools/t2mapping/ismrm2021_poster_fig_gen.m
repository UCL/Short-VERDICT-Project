% ismrm2021_poster_fig_gen
%
% Adapted from presentt2
% 


fstem = '/Users/davidatkinson/OneDrive - University College London/data/LWI/ISMRM2021' ;
% outstem = '/Users/davidatkinson/OneDrive - University College London/data/LWF_results' ;



% - - 
j=1 ;

d{j}.Name = 'INN363' ;
d{j}.FMET2   = fullfile(fstem,'INN363', 'DICOM', 'I22') ; % 1.5mm
d{j}.FT2     = fullfile(fstem,'INN363', 'DICOM', 'I16') ; % T2 fast only
% d{j}.Fdiff   = fullfile(fstem,'Reimagine','2010034', 'DICOM', 'I13') ; % b2000

% d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
% d{j}.Ddiff =  [6e3 18e3  1] ;
d{j}.DT2   =  [0 80e3 1] ;

d{j}.sl = 9 ; %z 74.6
d{j}.MET2Position = [55    53       75       75 ] ;
d{j}.bilwfmax = 0.5 ;
d{j}.Dlwfmax = 0.5 ;
% d{j}.outdir = fullfile(outstem, 'RECLAIM_2010034') ;

j=2 ;

d{j}.Name = 'INN372' ;
d{j}.FMET2   = fullfile(fstem,'INN372', 'DICOM', 'I23') ; % 1.5mm  501
d{j}.FT2     = fullfile(fstem,'INN372', 'DICOM', 'I17') ; % T2 fast only

d{j}.DT2   =  [0 80e3 1] ;

d{j}.sl = 5 ; % z11.01
d{j}.MET2Position = [55    53       75       75 ] ;
d{j}.bilwfmax = 0.5 ;
d{j}.Dlwfmax = 0.5 ;


j=3 ;

d{j}.Name = 'INN357' ;
d{j}.FMET2   = fullfile(fstem,'INN357', 'DICOM', 'I49') ; % 1.5mm
d{j}.FT2     = fullfile(fstem,'INN357', 'DICOM', 'I26') ; % T2 

d{j}.DT2   =  [0 45e3 1] ;

d{j}.sl = 12 ; % Z 69.89 - 72.34
d{j}.MET2Position = [55    53       75       75 ] ;
d{j}.bilwfmax = 0.5 ;
d{j}.Dlwfmax = 0.5 ;

% run
for j=1:3
    
    dinfoT2 = datparse(d{j}.FT2) ;
    [vT2, mT2] = d2mat(dinfoT2,{'slice'},'op','fp') ;
    
    dinfoMET2 = datparse(d{j}.FMET2) ;
    [vMET2, mMET2, lMET2] = d2mat(dinfoMET2, {'slice','echo'},'op','fp') ;
    
    
    [vcMET2, gcMET2] = dinplanet(vMET2, mMET2.geom, 'crop', d{j}.MET2Position) ;
    
    %[vcT2, gcT2] = dinplanet(vT2, mT2.geom, 'crop', d{j}.T2Position) ;
    [vcT2] = dreslice(vT2, mT2, gcMET2, 'PixelSpacing','input','sliceBoxKernel',true) ;
    
    
    IT2 = mat2gray(vcT2(:,:,d{j}.sl), d{j}.DT2(1:2) ) ;
    IT2 = imadjust(IT2, [],[],d{j}.DT2(3)) ;
    
    % bi-fit
    dfit = t2fitf(vcMET2,mMET2.effTEVec_indata / 1000) ;
    bilwf = zeros(dfit.szmap) ;
    bilwf(dfit.mloc) = dfit.lwf ;
    Ibilwf = mat2gray(bilwf(:,:,d{j}.sl),[0 d{j}.bilwfmax]) ;
  
% Old code. Changed to allow delayed mask so that starting estimate uses
% whole slice
%     [OUT,C,P] = LWI_Devine(vcMET2(:,:,d{j}.sl,:), mMET2.effTEVec_indata / 1000, 'waitdisp', true) ;
%     lwfDevine = OUT(:,:,1,9) ;
%     IDevine = mat2gray(lwfDevine,[0 d{j}.Dlwfmax]) ;

    dmask = zeros([size(vMET2,1) size(vMET2,2)]) ;
    yl = round(d{j}.MET2Position(2)) ;
    yu = yl + round(d{j}.MET2Position(4)) ;
             
    xl = round(d{j}.MET2Position(1)) ;
    xu = xl + round(d{j}.MET2Position(3)) ;
            
 % Put in draft ISMRM poster (5th April 2021)           
% %     rectind = {yl:yu, xl:xu} ; 
% %     dmask(rectind{:}) = 1 ;
% %     [OUT,C,P] = LWI_Devine(vMET2(:,:,d{j}.sl,:), mMET2.effTEVec_indata / 1000, ...
% %         'waitdisp', true, 'delayedmask', dmask) ;
% %     lwfDevine = OUT(rectind{:},1,9) ;
% %     IDevine = mat2gray(lwfDevine,[0 d{j}.Dlwfmax]) ;
   
                
%     binRMSE = zeros(dfit.szmap) ;
%     binRMSE(dfit.mloc) = dfit.nRMSE ;
%     IbinRMSE = mat2gray(binRMSE(:,:,d{j}.sl),[0 0.2]) ;
%     
%     % pd_bifixed
%     dfit = t2fitf(vcMET2(:,:,d{j}.sl,:),mMET2.effTEVec_indata / 1000, 'fmethod', 'pd_bifixed') ;
%     pdbilwf = zeros(dfit.szmap) ;
%     pdbilwf(dfit.mloc) = dfit.lwf ;
%     Ipdbilwf = mat2gray(pdbilwf,[0 d{j}.lwfmax]) ;
%     
%     pdbinRMSE = zeros(dfit.szmap) ;
%     pdbinRMSE(dfit.mloc) = dfit.nRMSE ;
%     IpdbinRMSE = mat2gray(pdbinRMSE,[0 0.2]) ;
%     
%     
    % NNLS
    dfit = t2fitf(vcMET2(:,:,d{j}.sl,:),mMET2.effTEVec_indata / 1000, 'fmethod', 'pd_NNLS') ;
    pdnnlslwf = zeros(dfit.szmap) ;
    pdnnlslwf(dfit.mloc) = dfit.lwf ;
    Ipdnnls = mat2gray(pdnnlslwf,[0 d{j}.Dlwfmax]) ;
%     
%     pdnnlsnRMSE = zeros(dfit.szmap) ;
%     pdnnlsnRMSE(dfit.mloc) = dfit.nRMSE ;
%     IpdnnlsnRMSE = mat2gray(pdnnlsnRMSE,[0 0.2]) ;
    
    
    
    figure('Name',d{j}.Name)
    t = tiledlayout(1,3,'Padding','none','TileSpacing','none') ;
    
    nexttile(1) ;
    imshow(IT2)
    nexttile(2)
    imshow(Ibilwf)
    
    nexttile
%    imshow(IDevine)
    imshow(Ipdnnls)

%     axbilwf = nexttile ;
%     imshow(Ibilwf) ; hold on
%     nexttile
%     imshow(Ipdbilwf)
%     nexttile
%     imshow(Ipdnnls)
%     nexttile(10)
%     imshow(IbinRMSE)
%     nexttile(11)
%     imshow(IpdbinRMSE)
%     nexttile(12)
%     imshow(IpdnnlsnRMSE)
    
    if isfield(d{j},'Flmask')
        S = load(d{j}.Flmask,'lmask') ;
        [lmask_crop, gcMET2] = dinplanet(S.lmask, mMET2.geom, 'crop', d{j}.MET2Position) ;
        [M,c] = contour(axbilwf,double(lmask_crop(:,:,d{j}.sl)),[0.5 0.5]) ;
        c.LineColor = [ 1 0 0 ] ;
        c.LineWidth = 2 ;
    end
    
    
end



