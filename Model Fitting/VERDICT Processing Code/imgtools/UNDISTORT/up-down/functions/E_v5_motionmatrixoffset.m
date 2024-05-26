function [u] = E_v5_motionmatrixoffset(v,arg1,arg2)
% Encoding function with dB0 phase term and coil sensitivities
% JH : 30/05/13

Sk1=E_v7_motionmatrix(v,arg1);
Sk2=E_v7_motionmatrix(v,arg2);
u=[Sk1;Sk2];
tic;toc %E_v3
end