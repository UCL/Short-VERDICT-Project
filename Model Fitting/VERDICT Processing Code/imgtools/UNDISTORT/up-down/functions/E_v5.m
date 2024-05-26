function [u] = E_v5(v,arg1,arg2)
% Encoding function with dB0 phase term and coil sensitivities


Sk1=E_v4(v,arg1);
Sk2=E_v4(v,arg2);
u=[Sk1;Sk2];
tic;toc 
end