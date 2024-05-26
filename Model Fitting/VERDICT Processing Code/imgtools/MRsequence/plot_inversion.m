function plot_inversion(T1s, ITR)
% plot_inversion
%
%
% Plot of Mz as a function of time for seqeunce with inversion pulses
% Can check null point against: 
% https://www.seichokai.or.jp/fuchu/dept1603.php
%
% D.Atkinson@ucl.ac.uk
%
%


figure('Name',['(plot_inversion) Mz vs time'])
title(['inv TR: ',num2str(ITR),' T1s: ',num2str(T1s)])
   
for it1 = 1:length(T1s)
    [Mz, times] = calc_inversion(T1s(it1), ITR) ;
    plot(times, Mz,'DisplayName',[num2str(T1s(it1)),'ms'])
    hold on, grid on
end

legend