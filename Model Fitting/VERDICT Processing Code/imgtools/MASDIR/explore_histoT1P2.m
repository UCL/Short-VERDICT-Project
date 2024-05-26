% Explore HISTO MRI patient T1 P2
% (24 June 2022)
%
% See Graeme Bydder Crib Sheet v3.6

ddir = '/Users/davidatkinson/Library/CloudStorage/OneDrive-UniversityCollegeLondon/data/20220624HISTO/Anon/DICOM' ;

dTIsTEs = dmfparse(fullfile(ddir,'I38'));
dTIsTEi = dmfparse(fullfile(ddir,'I40'));
dTIiTEs = dmfparse(fullfile(ddir,'I42'));
dTIiTEi = dmfparse(fullfile(ddir,'I44'));

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
dLWI = dmfparse(fullfile(ddir,'I26'));
[vlwi,mlwi] = d2mat(dLWI,{'slice','effTE'},'op','fp') ;

lwf = vlwi(:,:,:,8)./vlwi(:,:,:,1) ;

coords = {70:147, 72:160};

figure
tl= tiledlayout(4,5,"TileSpacing","tight","Padding","tight","TileIndexing","rowmajor") ;
for slice = 13:17
    num = tilenum(tl, 2, slice-12) ;
    nexttile(num)
    imshow(SIR(coords{:},slice), [-3000 0])

    num = tilenum(tl, 1, slice-12) ;
    nexttile(num)
    imshow(dSIR(coords{:},slice), [-1 0.25])

    num = tilenum(tl, 3, slice-12) ;
    nexttile(num)
    imshow(-T1(coords{:},slice), [1000 2000])

    num = tilenum(tl, 4, slice-12)
    nexttile(num)
    imshow(lwf(coords{:},slice-4), [0 0.5])

end
