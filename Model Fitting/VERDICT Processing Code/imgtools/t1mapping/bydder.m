
dt1 = getdinfo(dselector,'TIs') ;
ITYPE = 6 ; 
[vir, mir] = d2mat(dt1,{'InversionTime','itype'},'itype',ITYPE,'op','fp') ;

tis = num2str(mir.tiVec') ;

figure('Name','Inv time derived')
tiledlayout(2,3,'Padding','compact','TileSpacing','compact')
nexttile(1)
imshow(vir(:,:,6)-vir(:,:,5),[])
title([tis(6,:), ' - ',tis(5,:)])

nexttile(1+3)
imshow(vir(:,:,5)-vir(:,:,6),[])
title([tis(5,:), ' - ',tis(6,:)])

nexttile(2)
imshow(vir(:,:,5)-vir(:,:,4),[])
title([tis(5,:), ' - ',tis(4,:)])

nexttile(2+3)
imshow(vir(:,:,4)-vir(:,:,5),[])
title([tis(4,:), ' - ',tis(5,:)])

nexttile(3)
imshow(vir(:,:,6)-vir(:,:,4),[])
title([tis(6,:), ' - ',tis(4,:)])

nexttile(3+3)
imshow(vir(:,:,4)-vir(:,:,6),[])
title([tis(4,:), ' - ',tis(6,:)])

figure('Name','relative images')
tiledlayout(1,2,'Padding','compact','TileSpacing','compact')
nexttile
SIR = vir(:,:,4)-vir(:,:,5) ;
rSIR = vir(:,:,5)-vir(:,:,4) ;
AIR = vir(:,:,4)+vir(:,:,5) ;
dSIR = SIR./AIR ;
drSIR = rSIR ./ AIR ;

imshow(dSIR,[])
title('dSIR')

nexttile
imshow(drSIR,[])
title('drSIR')
