% testing scalings for ADCs

% Read in scanner ADCs
% Select slice and ROI
% Pixel value and Displayed Value

% Read in b-value images, calc ADC
% Check scale parameters throughout data.

disp(['Select scanner ADC'])
dsADC = datparse ;

[vsADC, msADC] = d2mat(dsADC,{'slice'},'op','dv') ;
eshow(vsADC, msADC)


disp(['Select diffusion images for ADC calc.'])
dsDWI = datparse ;

[vsDWI, msDWI] = d2mat(dsDWI,{'slice','bv'},'op','fp') ;
[ADC, S0] = calcADC(vsDWI, msDWI.bvVec) ;
eshow(ADC,msDWI)


