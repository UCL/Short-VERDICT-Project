function plot_humidity
% PLOT_HUMIDITY Plots humidity file from Philips monitoring of MR room.
% plot_humidity
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

refdate = '27-Feb-2013' ;

refdn = datenum(refdate) ;

disp(['Select Philips humidity file, e.g. G:\Site\monitor_System_HumExamRoom.dat'])
[humid, dnum] = humidity_Philips_read;

figure
plot(dnum-refdn, humid,'r')
hold on
grid
title('Philips - red, logger - blue')
xlabel(['Days since ',refdate])
ylabel(['Rel Humidity'])

lhumid = [0 0 0] ; % dummy
while length(lhumid) > 0
    disp(['Select humidity logging file'])
    [lhumid, ldnum] = humidity_logger_read ;
    
    plot(ldnum-refdn,lhumid,'b')
end



