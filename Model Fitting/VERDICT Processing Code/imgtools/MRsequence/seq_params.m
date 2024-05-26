function [theta_rad, phi_rad, TR, rfspoil] = seq_params(seqtype)
%  [theta_rad, phi_rad, TR] = seq_params

FA = 8 ;
rfspoil = 150 ;  % Philips RF spoiling increment - should be 150.

TR = 10 ;  % 5.7697 
% From Philips simulator
% % act_am_scale = [0 0.093 0.181 0.265 0.345 0.42 0.49 0.556 0.617 0.673 0.726 ...
% %     0.773 0.816 0.855 0.889 0.918 0.943 0.964 0.98 0.991 0.998 ...
% %     ones([1 24]) ] ; % from logging an RF act_am_scale

act_am_scale = [0 0.093 0.181 0.265 0.345 0.42 0.49 0.556 0.617 0.673 0.726 ...
    0.773 0.816 0.855 0.889 0.918 0.943 0.964 0.98 0.991 0.998 ...
    ones([1 500]) ] ; % from logging an RF act_am_scale


theta = FA * act_am_scale ;
theta_rad = d2r(theta) ;

np = length(theta) ;
p=[1:np];
phi = cumsum((p-1).* rfspoil) ;
phi_rad = d2r(phi) ;

