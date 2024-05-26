function [CBV, FM] = explore_DSC(vd, md, bolus)
%
% [CBV, FM] = explore_DSC(pd, md, bolus)
%
%     
% Example for PET MR data
%  dinfo = = datparse ;
%  [vd, md,ld] = d2mat(dinfo,{'slice','aqno'},'op','fp') ; 
%  [CBV, FM] = explore_DSC(vd, md, 4) ;
%
% David Atkinson


TE = md.echoTimeVec_indata;

if ndims(vd) ==2 && size(vd,2) == 1
    vd = vd' ;
end

pd = vd ; % vd is volume, keeping shape, pd will be strung out pixels


szpd = size(pd) ;
pd = reshape(pd,[prod(szpd(1:end-1)) szpd(end)]) ;

% used for masking for pixelwise calculations
thresh = multithresh(pd(:,2)) ; % Otsu threshold on 2nd dynamic
locs = find(pd(:,2) > 0.8*thresh) ;
nloc = length(locs) ;


% Boxerman ROI
sz = size(vd) ;
dimd = ndims(vd) ;
ndyn = size(vd,dimd) ;

vdmax = max(vd,[],dimd);
vdmin = min(vd,[],dimd);

vddiff = vdmax-vdmin ;
[h, roi] = roianal(vddiff,md,'ismodal',true) ;

figure('Name','ROI CTCs')
for iroi = 1:length(roi.slice)
    slcd = vd(:,:,roi.slice(iroi),:) ;
    BWrep = repmat(roi.BWc{iroi},[1 1 1 ndyn]) ;
    vdroi = slcd(BWrep) ;
    vdroir = reshape(vdroi,[length(vdroi)/ndyn  ndyn]) ;
    roid_mean = mean(vdroir,1) ;
    
    [mean_base, SD_base, bstart, bend] = DSC_baseline(roid_mean,'plotb',false) ;
    C = - log(roid_mean/mean_base)/TE ;
   
    if iroi ==1
        Cref = C(bend:end) ;
    else
        Cmain = C(bend:end) ;
    end
    
    plot(C), hold on, grid on
    lcc{iroi} = roi.labelc{iroi}{:} ;
end
legend(lcc)

if exist('Cmain','var')
  [K1roi, K2roi] = Boxerman(Cref, Cmain);
end

% Boxerman over whole image
K2_par = zeros([nloc 1]) ;
K1_par = zeros([nloc 1]) ;
CBVcorr_par = zeros([nloc 1]) ;

parfor iloc = 1:nloc
    [mean_base, SD_base, bstart, bend] = DSC_baseline(pd(locs(iloc),:),'plotb',false) ;
    C = - log(pd(locs(iloc),:)/mean_base)/TE ;
    [K1p, K2p, CBVcorrp] = Boxerman(Cref, C(bend:end));
    K1_par(iloc) = K1p ;
    K2_par(iloc) = K2p ;
    CBVcorr_par(iloc) = CBVcorrp ;
end
K1 = zeros(szpd(1:end-1)) ;
K2  = zeros(szpd(1:end-1)) ;
CBVcorr  = zeros(szpd(1:end-1)) ;

K1(locs) = K1_par ;
K2(locs) = K2_par ;
CBVcorr(locs) = CBVcorr_par ;

eshow(K1)
eshow(K2)
eshow(CBVcorr)

CBV =1 ; FM = 1 ;
return

% CBV, FM
CBV_par = zeros([nloc 1]) ;
FM_par  = zeros([nloc 1]) ;

parfor iloc = 1:nloc
    
  [mean_base, SD_base, bstart, bend] = DSC_baseline(pd(locs(iloc),:),'plotb',false) ;
  % pdm = squeeze(mean(pd,1)) ; % not needed here
  % C = - log(pdm/mean_base)/TE ;
   C = - log(pd(locs(iloc),:)/mean_base)/TE ;
   
   
  [pCBV, pFM] = gv_fit(C, bend, bolus) ;
  CBV_par(iloc) = pCBV ;
  FM_par(iloc) = pFM ;
end

CBV = zeros(szpd(1:end-1)) ;
FM  = zeros(szpd(1:end-1)) ;

CBV(locs) = CBV_par ;
FM(locs) = FM_par ;






    
    

