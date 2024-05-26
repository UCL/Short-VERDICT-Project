% Explore HISTO MRI patient T1 P3
% (06 July 2022)
%
% See Graeme Bydder Crib Sheet v3.6

ddir = ['/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/20220706_Histo_MR_recall/anon/DICOM'] ;

dTIsTEs = dmfparse(fullfile(ddir,'I36'));
dTIsTEi = dmfparse(fullfile(ddir,'I38'));
dTIiTEs = dmfparse(fullfile(ddir,'I40'));
dTIiTEi = dmfparse(fullfile(ddir,'I42'));

ITYPE = 6 ; % magnitude data
[vss,mss] = d2mat(dTIsTEs,{'slice','itype'},'itype',ITYPE,'op','fp') ;
[vsi,msi] = d2mat(dTIsTEi,{'slice','itype'},'itype',ITYPE,'op','fp') ;
[vis,mis] = d2mat(dTIiTEs,{'slice','itype'},'itype',ITYPE,'op','fp') ;
[vii,mii] = d2mat(dTIiTEi,{'slice','itype'},'itype',ITYPE,'op','fp') ;

TIs = mss.tiVec_indata ;
TIi = mis.tiVec_indata ;

SIR = vss - vis ;
SAIR = vss + vis ;
dSIR = SIR ./ SAIR ;

% Bydder Crib Sheet v3.6, eqn 19
T1 = ((TIs - TIi) * dSIR - (TIs+TIi)) / log(4) ;

% LWI
% dLWI = dmfparse(fullfile(ddir,'I26'));
% [vlwi,mlwi] = d2mat(dLWI,{'slice','effTE'},'op','fp') ;
% 
% lwf = vlwi(:,:,:,8)./vlwi(:,:,:,1) ;

% Luminal Water Ratio
dT2WREF = dmfparse(fullfile(ddir,'I60'));
dT2Whigh = dmfparse(fullfile(ddir,'I56')) ;
[vshort,mshort] = d2mat(dT2WREF,{'slice'},'op','fp') ;
[vlong, mlong] = d2mat(dT2Whigh,{'slice'},'op','fp') ;

vratio = vlong./vshort ;

vratioatT1 = vresample(vratio,mlong,mss) ;

% MS TI 1000
dMS1000 = dmfparse(fullfile(ddir,'I34'));
[vms1000,mms1000] = d2mat(dMS1000,{'slice','itype'},'itype',ITYPE,'op','fp') ;
vms1000atT1 = vresample(vms1000,mms1000,mss) ;

% T2W
dT2W = dmfparse(fullfile(ddir,'I32'));
[vT2W, mT2W] = d2mat(dT2W,{'slice'},'op','fp') ;
vT2WatT1 = vresample(vT2W,mT2W,mss) ;


coords = {65:147, 65:160};
sstart = 6 ; send = 19 ;

figure
tl= tiledlayout(4,(send-sstart+1),"TileSpacing","none","Padding","tight","TileIndexing","rowmajor") ;
for slice = sstart:send
    num = tilenum(tl, 1, slice-(sstart-1)) ;
    nexttile(num)
    imshow(vT2WatT1(coords{:},slice), [0 1500])
    if slice==sstart;ylabel('T2W');end

    num = tilenum(tl, 2, slice-(sstart-1)) ;
    nexttile(num)
    imshow(dSIR(coords{:},slice), [-1 0.25])
    if slice==sstart;ylabel('dSIR');end

%     num = tilenum(tl, 3, slice-(sstart-1)) ;
%     nexttile(num)
%     imshow(-T1(coords{:},slice), [1000 2000])

    num = tilenum(tl, 4-1, slice-(sstart-1)) ;
    nexttile(num)
    imshow(vratioatT1(coords{:},slice), [0 0.5])
    if slice==sstart;ylabel('ratio');end

    num = tilenum(tl, 5-1, slice-(sstart-1)) ;
    nexttile(num)
    imshow(vms1000atT1(coords{:},slice), [0 3000])
    if slice==sstart;ylabel('TI1000');end

end
