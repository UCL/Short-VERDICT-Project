function test_dreslice_data1
% TEST_DRESLICE_DATA1
% 
% Tests dreslice with data acqired on scanner at different agulations
%

disp(['Will reslice A into geometry of B'])

disp(['Select A (to be resliced)'])
dinfoA = datparse(dselect) ;

disp(['Select B (fixed)'])
dinfoB = datparse(dselect) ;

[vA,mA] = d2mat(dinfoA,{'slice'},'op','dv') ;
[vB,mB] = d2mat(dinfoB,{'slice'},'op','dv') ;

[volrs, mrs] = dreslice(vA, mA, mB) ;

eshow(vB,  mB,'vB')
eshow(volrs, mrs, 'rs')


