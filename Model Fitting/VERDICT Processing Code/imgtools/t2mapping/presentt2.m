% presentt2 PRESENT T2 results
% Also, can write Dicom files
%
%
% 


fstem = '/Users/davidatkinson/University College London/Gong, Yangcan - Luminal Water Imaging' ;
outstem = '/Users/davidatkinson/OneDrive - University College London/data/LWF_results' ;

j=1 ;

d{j}.Name = '2010034' ;
d{j}.FMET2   = fullfile(fstem,'Reimagine','2010034', 'DICOM', 'I17') ; % 1.5mm
d{j}.FT2     = fullfile(fstem,'Reimagine','2010034', 'DICOM', 'I11') ;
d{j}.Fdiff   = fullfile(fstem,'Reimagine','2010034', 'DICOM', 'I13') ; % b2000

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [6e3 18e3  1] ;
d{j}.DT2   =  [0 4e4 1] ;

d{j}.sl = 6 ;
d{j}.MET2Position = [60.729       71.243       70.257       70.257 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'RECLAIM_2010034') ;

% - - 
j=2 ;

d{j}.Name = '2010057' ;
d{j}.FMET2   = fullfile(fstem,'Reimagine','2010057', 'DICOM', 'I17') ; % 1.5mm
d{j}.FT2     = fullfile(fstem,'Reimagine','2010057', 'DICOM', 'I11') ;
d{j}.Fdiff   = fullfile(fstem,'Reimagine','2010057', 'DICOM', 'I13') ; % b2000

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 30e3  1] ;
d{j}.DT2   =  [0 42e3 1] ;

d{j}.sl = 9 ;
d{j}.MET2Position = [68.043       67.129       59.886       59.886 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'RECLAIM_2010057') ;

% - - 
j=3 ;

d{j}.Name = '2010026' ;
d{j}.FMET2   = fullfile(fstem,'Reimagine','2010026', 'DICOM', 'I20') ; % 1.5mm
d{j}.FT2     = fullfile(fstem,'Reimagine','2010026', 'DICOM', 'I14') ;
d{j}.Fdiff   = fullfile(fstem,'Reimagine','2010026', 'DICOM', 'I16') ; % b2000

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 40e3  1] ;
d{j}.DT2   =  [0 4e4 1] ;

d{j}.sl = 9 ;
d{j}.MET2Position = [61.643       56.614       71.314       71.314 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'RECLAIM_2010026') ;

% - - 
j=4 ;

d{j}.Name = '2010049' ;
d{j}.FMET2   = fullfile(fstem,'Reimagine','2010049', 'DICOM', 'I18') ; % 1.5mm
d{j}.FT2     = fullfile(fstem,'Reimagine','2010049', 'DICOM', 'I12') ;
d{j}.Fdiff   = fullfile(fstem,'Reimagine','2010049', 'DICOM', 'I14') ; % b2000

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 40e3  1] ;
d{j}.DT2   =  [0 4e4 1] ;

d{j}.sl = 11 ;
d{j}.MET2Position = [57.986       53.871       76.343       76.343 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'RECLAIM_2010049') ;

% - - 
j=5 ;

d{j}.Name = 'inn345' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn345sen','DICOM','I35') ; % 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn345sen','DICOM','I17') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn345sen','DICOM','I23') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn345sen','GB','1p5mm','1101','INN345SEN_8-echo_1p5mm17sl_1101.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 60e3  1] ;
d{j}.DT2   =  [0 7e4 1] ;

d{j}.sl = 6 ;
d{j}.MET2Position = [62.557       61.643       64.457       64.457 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'inn345') ;


% - - 
j=6 ;

d{j}.Name = 'inn348 (Motion between T2 and MET2?' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn348lga','1p5mm','I36') ; % 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn348lga','DICOM','I16') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn348lga','DICOM','I22') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn348lga','GB','1p5mm','1001','INN348LGA_8-echo_1p5mm17sl_1001.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [4e3 50e3  1] ;
d{j}.DT2   =  [0 7e4 1] ;

d{j}.sl = 6 ;
d{j}.MET2Position = [62.557       61.643       64.457       64.457 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'inn348') ;

% - - 
j=7 ;

d{j}.Name = 'inn354_6' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn354tcl','1p5mm','601','I26') ; % rpt is I28, 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn354tcl','DICOM','I18') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn354tcl','DICOM','I32') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn354tcl','1p5mm','601','inn354tcl_8-echo_1p5mm17sl_601.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 60e3  1] ;
d{j}.DT2   =  [0 7e4 1] ;

d{j}.sl = 6 ;
d{j}.MET2Position = [52.5       41.986       76.343       76.343 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'inn354_6') ;


% - - 
j=8 ;

d{j}.Name = 'inn360' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn360rbe','1p5mm','601','I25') ; % , 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn360rbe','DICOM','I19') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn360rbe','DICOM','I31') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn360rbe','1p5mm','601','INN360RBE_8-echo_1p5mm17sl_601.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [20e3 60e3  1] ;
d{j}.DT2   =  [0 8e4 1] ;

d{j}.sl = 7 ;
d{j}.MET2Position = [52.957       45.186       85.943       85.943 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'inn360') ;

% - - 
j=9 ;

d{j}.Name = 'inn363' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn363jjo','1p5mm','601','I24') ; % , 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn363jjo','DICOM','I16') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn363jjo','DICOM','I28') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn363jjo','1p5mm','601','INN363JJO_8-echo_1p5mm17sl_601.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 70e3  1] ;
d{j}.DT2   =  [0 8e4 1] ;

d{j}.sl = 9 ;
d{j}.MET2Position = [58.9       61.186         73.6         73.6 ] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'inn363') ;

% - - 
j=10 ;

d{j}.Name = 'inn364' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn364rba','1p5mm','901','I46') ; % , 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn364rba','DICOM','I23') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn364rba','DICOM','I40') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn364rba','1p5mm','901','INN364RBA_8-echo_1p5mm17sl_901.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 30e3  1] ;
d{j}.DT2   =  [0 5e4 1] ;

d{j}.sl = 8 ;
d{j}.MET2Position = [58.443       52.957       77.257       77.257] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'inn364') ;

% - - 
j=11 ;

d{j}.Name = 'inn365' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn365jwh','1p5mm','901','I46') ; % , 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn365jwh','DICOM','I23') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn365jwh','DICOM','I40') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn365jwh','1p5mm','901','INN365JWH_8-echo_1p5mm17sl_901.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 30e3  1] ;
d{j}.DT2   =  [0 5e4 1] ;

d{j}.sl = 9 ;
d{j}.MET2Position = [50.671       50.671       87.771       87.771] ;
d{j}.lwfmax = 0.4 ;
d{j}.outdir = fullfile(outstem, 'inn365') ;

% - - 
j=12 ;

d{j}.Name = 'inn378' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn378jcl','1p5mm','501','I24') ; % , 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn378jcl','DICOM','I18') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn378jcl','DICOM','I30') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn378jcl','1p5mm','501','INN378JCL_8-echo_1p5mm17sl_501.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 40e3  1] ;
d{j}.DT2   =  [0 8e4 1] ;

d{j}.sl = 11 ;
d{j}.MET2Position = [53.414         52.5       81.829       81.829] ;
d{j}.lwfmax = 0.4 ;

d{j}.outdir = fullfile(outstem, 'inn378') ;

% - - 
j=13 ;

d{j}.Name = 'inn376' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn376bfu','1p5mm','501','I23') ; % , 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn376bfu','DICOM','I17') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn376bfu','DICOM','I29') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn376bfu','1p5mm','501','INN376BFU_8-echo_1p5mm17sl_501.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [10e3 70e3  1] ;
d{j}.DT2   =  [0 8e4 1] ;

d{j}.sl = 7 ;
d{j}.MET2Position = [55.243       51.129       84.114       84.114] ;
d{j}.lwfmax = 0.4 ;

d{j}.outdir = fullfile(outstem, 'inn376') ;

% - - 
j=14 ;

d{j}.Name = 'inn357' ;
d{j}.FMET2 = fullfile(fstem,'INNOVATE for David','inn357rme','1p5mm','1001','I49') ; % , 1.5mm
d{j}.FT2   = fullfile(fstem,'INNOVATE for David','inn357rme','DICOM','I26') ; % T2
d{j}.Fdiff = fullfile(fstem,'INNOVATE for David','inn357rme','DICOM','I43') ; % diff

d{j}.Flmask =  fullfile(fstem,'INNOVATE for David','inn357rme','1p5mm','1001','INN357RME_8-echo_1p5mm17sl_1001.mat') ;

d{j}.Dlast3 = [ 0 0.4  1 ] ;  % disp min, max, gamma
d{j}.Ddiff =  [3e3 11e3  1] ;
d{j}.DT2   =  [0 5e4 1] ;

d{j}.sl = 10 ;
d{j}.MET2Position = [62.1       62.557       63.086       63.086] ;
d{j}.lwfmax = 0.4 ;

d{j}.outdir = fullfile(outstem, 'inn357') ;



% run
for j=14:14
    if isfield(d{j},'outdir')
        iswrite = true;
    else 
        iswrite = false ;
    end
    
    %iswrite = false 
    
    if iswrite
        if ~exist(d{j}.outdir,'dir')
            mkdir(d{j}.outdir)
        end
        copyfile(d{j}.FT2, d{j}.outdir) ;
        copyfile(d{j}.Fdiff, d{j}.outdir) ;
    end
    
    
    dinfoT2 = datparse(d{j}.FT2) ;
    [vT2, mT2] = d2mat(dinfoT2,{'slice'},'op','fp') ;
    
    dinfoMET2 = datparse(d{j}.FMET2) ;
    [vMET2, mMET2, lMET2] = d2mat(dinfoMET2, {'slice','echo'},'op','fp') ;
    
    % DICOM files
    % Do whole volume for last3 and bi-fixed
    dfit = t2fitf(vMET2, mMET2.effTEVec_indata / 1000, 'fmethod', 'last3') ;
    vl3 = zeros(dfit.szmap) ; vl3(dfit.mloc) = dfit.lwf ;
    
    if iswrite
        writeDicom(vl3, 'LWF_2', 'geom', mMET2.geom, 'FrameOfReferenceUID', 'keep', ...
            'header', {dinfoMET2, lMET2}, 'ImageComments','last3', ...
            'SeriesDescription','last3 (x1000)', 'SeriesNumber', 523, 'SeriesInstanceUID','new', ...
            'ProtocolName','last3 (x1000)','folder_name', d{j}.outdir, 'fnstem', 'last3')
    end
    
    % bi-fit
    dfit = t2fitf(vMET2,mMET2.effTEVec_indata / 1000) ;
    bilwf = zeros(dfit.szmap) ;
    bilwf(dfit.mloc) = dfit.lwf ;
    
    if iswrite 
        writeDicom(bilwf, 'LWF_2', 'geom', mMET2.geom, 'FrameOfReferenceUID', 'keep', ...
        'header', {dinfoMET2, lMET2}, 'ImageComments','bi-fixed', ...
        'SeriesDescription','bi-fixed  (x1000)', 'SeriesNumber', 524,'SeriesInstanceUID','new', ...
        'ProtocolName','bi-fixed (x1000)', 'folder_name', d{j}.outdir, 'fnstem', 'bi-fixed')
    end
    
    
    dinfodiff = datparse(d{j}.Fdiff) ;
    if j>=5
        [vdiff, mdiff] = d2mat(dinfodiff,{'slice','ddty'},...
            'ddty',2,'op','fp') ;
    else
        [vdiff, mdiff] = d2mat(dinfodiff,{'slice'},'op','fp') ;
    end
    
    [vcMET2, gcMET2] = dinplanet(vMET2, mMET2.geom, 'crop', d{j}.MET2Position) ;
    
    %[vcT2, gcT2] = dinplanet(vT2, mT2.geom, 'crop', d{j}.T2Position) ;
    [vcT2]     = dreslice(vT2, mT2, gcMET2, 'PixelSpacing','input','sliceBoxKernel',true) ;
    [vcdiff, mcdiff] = dreslice(vdiff, mdiff, gcMET2,'sliceBoxKernel',true) ;
    
    % use t2fitf for last3
    dfit = t2fitf(vcMET2(:,:,d{j}.sl,:), mMET2.effTEVec_indata / 1000, 'fmethod', 'last3') ;
    vl3 = zeros(dfit.szmap) ; vl3(dfit.mloc) = dfit.lwf ;
    
    Il3 = mat2gray(vl3, d{j}.Dlast3(1:2) ) ;
    Il3 = imadjust(Il3,[],[],d{j}.Dlast3(3)) ;
    
    Idiff = mat2gray(vcdiff(:,:,d{j}.sl), d{j}.Ddiff(1:2) ) ;
    Idiff = imadjust(Idiff,[],[], d{j}.Ddiff(3)) ;
    
    IT2 = mat2gray(vcT2(:,:,d{j}.sl), d{j}.DT2(1:2) ) ;
    IT2 = imadjust(IT2, [],[],d{j}.DT2(3)) ;
    
    % bi-fit
    dfit = t2fitf(vcMET2,mMET2.effTEVec_indata / 1000) ;
    bilwf = zeros(dfit.szmap) ;
    bilwf(dfit.mloc) = dfit.lwf ;
    Ibilwf = mat2gray(bilwf(:,:,d{j}.sl),[0 d{j}.lwfmax]) ;
    
    binRMSE = zeros(dfit.szmap) ;
    binRMSE(dfit.mloc) = dfit.nRMSE ;
    IbinRMSE = mat2gray(binRMSE(:,:,d{j}.sl),[0 0.2]) ;
    
    % pd_bifixed
    dfit = t2fitf(vcMET2(:,:,d{j}.sl,:),mMET2.effTEVec_indata / 1000, 'fmethod', 'pd_bifixed') ;
    pdbilwf = zeros(dfit.szmap) ;
    pdbilwf(dfit.mloc) = dfit.lwf ;
    Ipdbilwf = mat2gray(pdbilwf,[0 d{j}.lwfmax]) ;
    
    pdbinRMSE = zeros(dfit.szmap) ;
    pdbinRMSE(dfit.mloc) = dfit.nRMSE ;
    IpdbinRMSE = mat2gray(pdbinRMSE,[0 0.2]) ;
    
    
    % NNLS
    dfit = t2fitf(vcMET2(:,:,d{j}.sl,:),mMET2.effTEVec_indata / 1000, 'fmethod', 'pd_NNLS') ;
    pdnnlslwf = zeros(dfit.szmap) ;
    pdnnlslwf(dfit.mloc) = dfit.lwf ;
    Ipdnnls = mat2gray(pdnnlslwf,[0 d{j}.lwfmax]) ;
    
    pdnnlsnRMSE = zeros(dfit.szmap) ;
    pdnnlsnRMSE(dfit.mloc) = dfit.nRMSE ;
    IpdnnlsnRMSE = mat2gray(pdnnlsnRMSE,[0 0.2]) ;
    
    
    
    if iswrite
        writeDicom(pdbilwf, 'LWF_2', 'geom', gcMET2(d{j}.sl), 'FrameOfReferenceUID', 'keep', ...
            'header', {dinfoMET2, lMET2(d{j}.sl,1)}, 'ImageComments','pdbifixed', ...
            'SeriesDescription','pd_bifixed  (x1000)', 'SeriesNumber', 525,'SeriesInstanceUID','new', ...
            'ProtocolName','pd_bifixed (x1000)', 'folder_name', d{j}.outdir, 'fnstem', 'pd_bifixed')
    
        writeDicom(pdnnlslwf, 'LWF_2', 'geom', gcMET2(d{j}.sl), 'FrameOfReferenceUID', 'keep', ...
            'header', {dinfoMET2, lMET2(d{j}.sl,1)}, 'ImageComments','pdnnls', ...
            'SeriesDescription','pdnnls  (x1000)', 'SeriesNumber', 526,'SeriesInstanceUID','new', ...
            'ProtocolName','pdnnls (x1000)', 'folder_name', d{j}.outdir, 'fnstem', 'pdnnls')
    end
    
    
    
    figure('Name',d{j}.Name)
    t = tiledlayout(2,6,'Padding','none','TileSpacing','none') ;
    nexttile(1) ;
    imshow(IT2)
    nexttile(2)
    imshow(Idiff)
    nexttile
    imshow(Il3)
    axbilwf = nexttile ;
    imshow(Ibilwf) ; hold on
    nexttile
    imshow(Ipdbilwf)
    nexttile
    imshow(Ipdnnls)
    nexttile(10)
    imshow(IbinRMSE)
    nexttile(11)
    imshow(IpdbinRMSE)
    nexttile(12)
    imshow(IpdnnlsnRMSE)
    
    if isfield(d{j},'Flmask')
        S = load(d{j}.Flmask,'lmask') ;
        [lmask_crop, gcMET2] = dinplanet(S.lmask, mMET2.geom, 'crop', d{j}.MET2Position) ;
        [M,c] = contour(axbilwf,double(lmask_crop(:,:,d{j}.sl)),[0.5 0.5]) ;
        c.LineColor = [ 1 0 0 ] ;
        c.LineWidth = 2 ;
    end
    
    
end



