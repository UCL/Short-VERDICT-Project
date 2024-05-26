function wrap_t2fit

dinfo = datparse(dselect) ;

[data, md] = d2mat(dinfo,{'slice','echo'},'op', 'fp') ;

% TO DO / ISSUES
% !! Probable bug with Milan data and plotting of signal - fit and scales
% look wrong.
%
%  Need for "official" version for XNAT - also good publicity
%  Devine has bugs in starting estimates, that sometimes 'cancel'
%  Devine starting estimate will vary with size of data passed in as it
%  uses median of slice - makes repeatability hard.
%
%  LWF depends on luminal contribution but their contributin to signal
%  depends on T2 - trade offs that can lead to problems. Merit in fixed
%  T2s.
%
% 1) Central repo
%     Fox wrong signal displayed if slice chnaged.
%     Devine method, corrected Devine, alternatives,
%     Test cases. Note Milan fit goes wrong for fixed-bi. (sometimes)
%     Core code that is free from manual input/output. Allows batching.
%     For b-fit fixed, produce montage of volume.
%    
% 2) Evaluate a selection of test cases. ReImagine, Milan
%
% 3) Nice to have
%     Better DICOM selection
%     Better colormap slider choice
%     Signal axis ylim adjustable. Smaller Disp max step sizes.
%     Fit quality threshold
%
%
% 4) Grant
%  Investigate using bladder or muscle or fat as means to estimate flip
%  angle.
%  Coping with "Milan" data that causes a poor fit.
%  Use signal analysis (slope, last points, known SNR) to set starting
%  estimates.
%  EPG modelling.
%  logspace for prob dist?
%  maybe add other NNLS
%  colormap - adjustments
%


%datin = data(188:358,160:360,10,:) ;
datin = data(288:358,260:360,10,:) ;
% datin = data(200:400,200:400,11,:) ;

%  [OUT,C,P] = LWI_Devine(double(datin), md.echoTimeVec_indata/1000) ;
%  eshow(OUT(:,:,:,9),'Name','OUT9')
%   
%  return
 
ft = t2bifit(datin, md.echoTimeVec_indata) ;



% bi   (exp2) fits  	Y = a*exp(b*x)+c*exp(d*x)
for iloc=1:ft.nloc
    biratio(ft.loc(iloc)) = ft.fbi{iloc}.c / (ft.fbi{iloc}.a + ft.fbi{iloc}.c)  ; 
end

eshow(datin(:,:,:,3),'Name','datin')

eshow(biratio)

% 
while true
rcs = input('Enter [row col slice]');

figure('Name',[num2str(rcs)]) ;
indinfull = sub2ind(ft.szmap, rcs(1), rcs(2), rcs(3)) ;
mloc = find(ft.loc == indinfull) ;
if isempty(mloc)
    warning(['Not in computed region'])
end

m = 2 ; n = 1 ;
% X = ft.flwi{mloc}.X ;
% 
% pr = ft.pr(X) ;
% subplot(m,n,1), plot(ft.prt2s, pr) 

%title(['LWF: ',num2str(ft.flwi{mloc}.LWF) ])
subplot(m,n,2)
plot(ft.fbi{mloc}, md.echoTimeVec_indata/1000, squeeze(datin(rcs(1),rcs(2),rcs(3),:)))
% plot(md.echoTimeVec_indata, ft.A*pr, 'DisplayName','LWI fit'),
grid on
end



