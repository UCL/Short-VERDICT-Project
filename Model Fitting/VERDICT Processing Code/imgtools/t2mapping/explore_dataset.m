function explore_dataset

% fn = '/Volumes/DA Windows/data/INNOVATE03DEC2018/DICOM/IM_0004' ;
% fn = '/Volumes/DA Windows/data/INNOVATE-REP-T2-B0/DICOM/I33_WIP_MS32echo_SENSE_1501' ;
dinfo = dmfparse(dselect);
[v,m,l] = d2mat(dinfo,{'slice','echo'},'op','fp') ;
ect = unique([dinfo.EffectiveEchoTime]) ;

def_slc = floor(size(v,3)/2) + 1 ;
figure
imshow4(v,def_slc,1,[0 max(v(:))*0.9])

slc = input('Enter slice number: ') ;

% For INNOVATE03DEC2018/DICOM/IM_0004 only
% ic = 0 ; lab = {}; coord = {};
% % [row column]
% ic = ic+1; coord{ic} = [96 56] ; lab{ic} = 'muscle left' ;
% ic = ic+1; coord{ic} = [96 141] ; lab{ic} = 'muscle right' ;
% ic = ic+1; coord{ic} = [88 80] ; lab{ic} = 'PZ bright' ;
% ic = ic+1; coord{ic} = [91 109] ; lab{ic} = 'PZ med' ;
% ic = ic+1; coord{ic} = [149 117] ; lab{ic} = 'fat' ;
% ic = ic+1; coord{ic} = [85 176] ; lab{ic} = 'femur' ;
% nc = ic ;
% 
% figure
% for ic = 1:nc
%     plot(ect, squeeze(v(coord{ic}(1),coord{ic}(2),slc,:)),'DisplayName',lab{ic}), hold on
% end
% grid on
% legend

eshow(v(:,:,slc,:),'Name',['slc ',num2str(slc)])

% Draw Muscle ROI
figure('Name','Draw muscle ROI')
img = v(:,:,slc,1) ;
imshow(img,[0 max(img(:))*0.9 ])

ha = gca ;
hroi_musc = drawpolygon('FaceAlpha',0) ;
bw_musc = createMask(hroi_musc) ;
loc_highe = find(ect>200) ;

sig = [] ;
for ie = loc_highe
    img = squeeze(v(:,:,slc,ie)) ;
    sig = [sig img(bw_musc).'] ;
end

noisem = median(sig) ;
noisestd = std(sig) ;
noise_thresh = noisem + 1*noisestd ;
disp(['Threshold is ',num2str(noise_thresh),' (mean ',num2str(noisem), ...
    ', std ',num2str(noisestd),')'])

% disp(['Draw FAT ROI'])
% hroi_fat = drawpolygon('FaceAlpha',0) ;
% bw_fat = createMask(hroi_fat) ;

% Fit a mono-exponential T2 to give an absolute scaling
loc_lowe = find(ect<160) ;
sigl = zeros(length(loc_lowe),1) ;
for ie = loc_lowe
    img = squeeze(v(:,:,slc,ie)) ;
    sigl(ie,1) = median(img(bw_musc)) ;
end
    
f = fit(ect(loc_lowe).',sigl(loc_lowe,1)-noisem,'exp1');

figure
ets = [0:1000];
pure = noisem + f.a*exp(ets*f.b) ; % f.b is negative
plot(ets, pure), hold on, grid on
plot(ect(loc_lowe),sigl(loc_lowe),'ro')
title(['T2: ',num2str(-1/f.b),' ms, a = ',num2str(f.a)])

% Set a reference intensity.
% Here choose muscle signal at 50ms.
sig_ref = noisem + f.a*exp(50*f.b) ;
disp(['Reference signal level: ',num2str(sig_ref)])



vmetric = zeros(size(v,1),size(v,2),size(v,3)) ;


% make image from all echoes greater than 200
vrgb = zeros(size(v,1),size(v,2),size(v,3),3) ;
for islc = 1:size(v,3) 
    mimg = sum(v(:,:,islc,loc_highe),4) / length(loc_highe) ;
    locnlw = find(mimg<(noise_thresh)) ;
    mat1 = double(v(:,:,islc,1)) ;
    im_gr = mat2gray(mat1,[0 max(mat1(:))*0.9 ]) ;
    [im_ind, cm] = gray2ind(im_gr) ;
    im_rgb = ind2rgb(im_ind, cm) ;
    redc = im_rgb(:,:,1) ;
    redc(locnlw) = 1 ;
    im_rgb(:,:,1) = redc ;
    vrgb(:,:,islc,:) = reshape(im_rgb,[size(im_rgb,1) size(im_rgb,2) 1 3]) ;
    
    met = (mimg - noise_thresh)/sig_ref ;
    vmetric(:,:,islc) = (-1/0.2)*met + 1 ;
    
%     Argb = im_rgb.^6 ;
%     Argb(:,:,2)=0 ;
%     Argb(:,:,3)=0 ;
%     
%     B = v(:,:,islc,1) ;
%     
%     figure
%     imshowpair(Argb,B,'blend')
%     disp(' ')
    
end

eshow(vrgb,'isrgb',true)
eshow(vmetric)
    



