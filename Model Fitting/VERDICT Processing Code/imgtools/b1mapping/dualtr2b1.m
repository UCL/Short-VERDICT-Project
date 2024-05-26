function dualtr2b1(dinfo, series_no)
% DUALTR2B1 From Philips single-frame data
%
% dualtr2b1(dinfo, series_no)
%

[vb1, mb1] = d2mat(dinfo,{'slice','itype'},'itype',[5 6],'series',series_no,'op','fp') ;
% Seems to read in the lower TR first.

% Yarnykh  AFI  MRM 57p192
% r = S2 / S1
% n = TR2 / TR1

r = vb1(:,:,:,2) ./ vb1(:,:,:,1) ;

n = mb1.RepetitionTime_Vec(2) / mb1.RepetitionTime_Vec(1) ;

fa_nom = mb1.faVec_indata ;

fa_act = 360/2/pi* acos( (r*n - 1)./(n-r) ) ;

B1 = fa_act / fa_nom ;

eshow(B1)

