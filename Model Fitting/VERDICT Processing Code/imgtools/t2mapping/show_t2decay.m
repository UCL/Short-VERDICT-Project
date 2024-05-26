function [pargs, ha] = show_t2decay
% SHOW_T2DECAY
%
%  [pargs, ha] = show_t2decay
%  
% See also ADD_POLY  POLY_PROFILES
%


% dfolder = '/Users/davidatkinson/OneDrive - University College London/data/LWI20190813' ;
% % pfolder = '32-echo malignant/inn104rwb/DICOM' ;
% % fn = 'I23' ; slc = 5 ; % right PZ should be slice 3
% 
% pfolder = '32-echo malignant/inn118cro/DICOM' ;
% fn = 'I20' ; slc = 2 ; % right PZ should be slice 2
% 
% ffn = fullfile(dfolder, pfolder, fn) ;
% 
% 
% dinfo = dmfparse(ffn);

dinfo = dmfparse(dselect);

[v,m,l] = d2mat(dinfo,{'slice','echo'},'op','fp') ;

def_slc = floor(size(v,3)/2) + 1 ;
figure
imshow4(v,def_slc,1,[0 max(v(:))*0.8])

slc = input('Enter slice number for ROIs ') ;

ect = unique([dinfo.EffectiveEchoTime]) ;

himf = figure ;
imshow(squeeze(v(:,:,slc,2)),[])
ha = gca ;


add_poly(ha) ;

poly_profiles(himf, slc, ect, v)
pargs = {himf, slc, ect, v} ;

disp(['To update signal plot'])
disp('poly_profiles(pargs{:})')
disp(' ')
disp('To add a polygon')
disp('add_poly(ha)')

return



% coords in x y (ie column row) for plot
ic = 0 ; lab = {}; coord = {};
ic = ic+1; coord{ic} = [139 106] ; lab{ic} = 'muscle left' ;
ic = ic+1; coord{ic} = [82 105 ] ; lab{ic} = 'PZ tumour' ;
ic = ic+1; coord{ic} = [52  106] ; lab{ic} = 'muscle right' ;
ic = ic+1; coord{ic} = [112 96] ; lab{ic} = 'PZ left' ;



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

nc = ic ;
figure('Name',fullfile(pfolder,fn))
for ic = 1:nc
    plot(ect, squeeze(v(coord{ic}(1),coord{ic}(2),slc,:)),'DisplayName',lab{ic}), hold on
end
grid on, xlabel('TE (ms)')
legend

figure
imshow(squeeze(v(:,:,slc,2)),[]), hold on
for ic = 1:nc
    plot(coord{ic}(1),coord{ic}(2),'+','MarkerSize',10,'DisplayName',lab{ic}), hold on
end


return




eshow(v(:,:,slc,:),'Name',['slc ',num2str(slc)])

figure
imshow4(v,slc,1,[0 max(v(:))*0.9])

% Draw Muscle ROI
figure('Name','Draw muscle ROI')
imshow(v(:,:,slc,1),[])
ha = gca ;
hroi = drawpolygon('FaceAlpha',0) ;
bw = createMask(hroi) ;
loc_highe = find(ect>200) ;

sig = [] ;
for ie = loc_highe
    img = squeeze(v(:,:,slc,ie)) ;
    sig = [sig img(bw).'] ;
end

noisem = median(sig) ;
noisestd = std(sig) ;
disp(['Threshold is ',num2str(noisem+2*noisestd)])


% make image from all echoes greater than 200
vrgb = zeros(size(v,1),size(v,2),size(v,3),3) ;
for islc = 1:size(v,3) 
    mimg = sum(v(:,:,islc,loc_highe),4) / length(loc_highe) ;
    locnlw = find(mimg<(noisem+1*noisestd)) ;
    im_gr = mat2gray(v(:,:,islc,1)) ;
    [im_ind, cm] = gray2ind(im_gr) ;
    im_rgb = ind2rgb(im_ind, cm) ;
    redc = im_rgb(:,:,1) ;
    redc(locnlw) = 1 ;
    im_rgb(:,:,1) = redc ;
    vrgb(:,:,islc,:) = reshape(im_rgb,[size(im_rgb,1) size(im_rgb,2) 1 3]) ;
end

eshow(vrgb,'isrgb',true)

    



