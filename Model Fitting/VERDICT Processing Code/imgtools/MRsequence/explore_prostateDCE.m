
% explore_prostateDCE

[theta_rad, phi_rad, TR, rfspoil] = seq_params ;

T2 = 80 ;

T1s = [700 1000 1400] ; 

nom_theta_rad = theta_rad(end) ; % nominal theta radians (after transition)

ssSI = ssSPGR(nom_theta_rad, TR, T1s) ;

theta_rad(200) = pi ; phi_rad(200) = 0 ;
theta_rad(300) = pi ; phi_rad(300) = 0 ;
theta_rad(400) = pi ; phi_rad(400) = 0 ;
theta_rad(500) = pi ; phi_rad(500) = 0 ;


figure
for it1 = 1:length(T1s)
    [F0,Fn,Zn,F] = EPG_GRE_TFE(theta_rad,phi_rad,TR,T1s(it1),T2) ;
    plot(abs(F0), 'DisplayName', num2str(T1s(it1)))
    grid on, hold on
    plot([0 length(F0)],[ssSI(it1) ssSI(it1)], 'DisplayName', ['ss: ',num2str(T1s(it1))] )
end

legend
title(['TR: ',num2str(TR),' FA(last): ',num2str(nom_theta_rad*360/2/pi), ' rfspoil: ',num2str(rfspoil) ])
axis([0 length(F0) 0 0.2])

