function t2sig

T1s = [1400 1400] ;
T2s = [60 300] ;

nprof = 100 ;

f1 = phantom([1 0.4 0.8 0 0 0 ], nprof) ;
f2 = phantom([1 0.1 0.2 0 0 30], nprof) ;

lwf = 0.2 ;
img1 = f1-lwf*f2;
img2 = f2*lwf ;


% compute echo signals from each T2 component separately
% simulate image at time of each echo as linear combination of tissues 
% find k-space for that readout
% loop over all echos (k-space lines)

% FT back to image domain



% s1 should be array with length nprof
[sigec, korder] = prostate_TSE(T1s, T2s) ; % with appropriate calling arguments.
iseq = 1 ;
s1=[] ; s2=[] ;
for ishot = 1:size(sigec,1)
    s1 = cat(1,s1,sigec{ishot, iseq, 1}) ;
    s2 = cat(1,s2,sigec{ishot, iseq, 2}) ;
end


kn = zeros(size(img1)) ;

for iprof = 1:nprof
    imgp1 = s1(iprof)*img1 ;
    imgp2 = s2(iprof)*img2 ;
    
    kp = k2i(imgp1+imgp2) ;
    kline = korder(iprof) ;
    kn(kline,:) = kp(kline,:) ;
end

ifinal = ik2(kn) ;
eshow(ifinal)
