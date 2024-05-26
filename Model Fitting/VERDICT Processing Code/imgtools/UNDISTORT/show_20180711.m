function show_20180711(varargin)
%SHOW_20180711 
%   '

% 'clinical' and 'research' scans

% Load clinical T2

clinfold = ['/Users/davidatkinson/data/UNDISTORT/20180711_UND027/', ...
    'DICOMS_anon/clinical_scan/export/DICOM/UND_027 [UND_027]/',... 
    '20180711 120849 [ - MR Prostate]'] ;

resfold = ['/Users/davidatkinson/data/UNDISTORT/20180711_UND027/', ...
    'DICOMS_anon/research_scan/export/DICOM/UND_027 [UND_027]/',... 
    '20180711 110915 [ - 3T000147A]'] ;

T2axfn = fullfile(clinfold,'Series 401 [MR - T2W TSE ax]', ...
    '1.3.6.1.4.1.5962.99.1.2320053295.2111626215.1531328443439.79.0.dcm') 


if~exist(T2axfn,'file')
    disp(['File does not exist'])
end
dinfT2ax = datparse(T2axfn) ;
[vT2,mT2,lT2] = d2mat(dinfT2ax,{'slice'},'op','fp') ;



% Diffusion clinical (not acquired with diffusion O images in DB)
difffn = fullfile(clinfold, 'Series 601 [MR - DWI 0 150 500 1000]', ...
    '1.3.6.1.4.1.5962.99.1.2320053295.2111626215.1531328443439.178.0.dcm' ) 

dinfodiff = datparse(difffn) ;
[vd,md,ld] = d2mat(dinfodiff,{'slice','bv'},'bv',0,'op','fp') ;
eshow(vd,md)

% 
[vdbv,mdbv,ldbv] = d2mat(dinfodiff,{'slice','bv'}, 'op','fp' ) ;
[ADC, S0] = calcADC(vdbv, mdbv.bvVec) ;
eshow(ADC, mdbv)

% T2W at diffusion 
[vT2atdiff, mat] = dreslice(vT2,mT2,md,'PixelSpacing','input') ;
eshow(vT2atdiff, mat)

% Diffusion research ('research' scans taken with Diffusion O images in DB) SPAIR
diffSPAIR = fullfile(resfold,'Series 901 [MR - DWIv2 0SPAIR 0 150 500 1000]', ...
    '1.3.6.1.4.1.5962.99.1.2320053295.2111626215.1531328443439.2119.0.dcm' ) ;

dinfodSPAIR = datparse(diffSPAIR) ;
[vdSPAIR,mdSPAIR,ldSPAIR] = d2mat(dinfodSPAIR,{'slice','bv'},'bv',0,'op','fp') ;
eshow(vdSPAIR,mdSPAIR)

% Problems with DiffusionDirectionality being 0 for b=0 and 2 for isotropic
% DWI
[vb,mb,lb] = d2mat(dinfodSPAIR, {'slice','bv','ddty'},'bv',mdSPAIR.bvVec_indata(2:end),'ddty',2,'op','fp') ;

v = cat(4,vdSPAIR, vb) ;
bvec = [mdSPAIR.bvVec mb.bvVec] ;
[ADCSPAIR, S0] = calcADC(v,bvec) ;
eshow(ADCSPAIR,mb)


% Diffusion research SPIR
diffSPIR = fullfile(resfold,'Series 1101 [MR - DWIv2 0SPIR 0 150 500 1000]', ...
    '1.3.6.1.4.1.5962.99.1.2320053295.2111626215.1531328443439.2399.0.dcm' ) ;

dinfodSPIR = datparse(diffSPIR) ;
%[vdSPIR,mdSPIR,ldSPIR] = d2mat(dinfodSPIR,{'slice','bv'},'bv',0,'op','fp') ;

% b1000
%[vdSPIRb1000,mdSPIRb1000,ldSPIRb1000] = d2mat(dinfodSPIR,{'slice','bv','ddty'},'bv',1000,'ddty',2,'op','fp') ;
%eshow(vdSPIRb1000, mdSPIRb1000)

%[vdb1000,mdb1000,ldb1000] = d2mat(dinfodiff,{'slice','bv','ddty'},'bv',1000,'ddty',2,'op','fp') ;
%eshow(vdb1000, mdb1000)

[vdSPIR,mdSPIR,ldSPIR] = d2mat(dinfodSPIR,{'slice','bv'},'bv',0,'op','fp') ;
%eshow(vdSPIR,mdSPIR)

% Problems with DiffusionDirectionality being 0 for b=0 and 2 for isotropic
% DWI
[vb,mb,lb] = d2mat(dinfodSPIR, {'slice','bv','ddty'},'bv',mdSPIR.bvVec_indata(2:end),'ddty',2,'op','fp') ;

v = cat(4,vdSPIR, vb) ;
bvec = [mdSPIR.bvVec mb.bvVec] ;
[ADCSPIR, S0] = calcADC(v,bvec) ;
eshow(ADCSPIR,mb)

% short TR with a b0 and b50
diffsTR = fullfile(resfold,'Series 1301 [MR - DWIshortTR 0 50 150 500 1000]', ...
    '1.3.6.1.4.1.5962.99.1.2320053295.2111626215.1531328443439.2830.0.dcm') ;
dinfosTR = datparse(diffsTR) ;
[vsTR0, msTR0] = d2mat(dinfosTR,{'slice','bv'},'bv',0,'op','fp') ;
[vsTRbv, msTRbv] = d2mat(dinfosTR, {'slice','bv','ddty'},'bv',msTR0.bvVec_indata(2:end),'ddty',2,'op','fp') ;

v = cat(4, vsTR0, vsTRbv) ;
bvec = [msTR0.bvVec msTRbv.bvVec] ;
[ADCsTRall, S0] = calcADC(v,bvec) ;
[ADCsTRn0,S0]=calcADC(v,bvec, bvec(2:end)) ;

eshow(cat(2,ADCsTRall, ADCsTRn0), msTR0)



end

