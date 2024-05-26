function plot_temperature
% PLOT_TEMPERATURE Plot Philips temperature file for monitoring MR room.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% See also PLOT_HUMIDITY
%

refdate = '01-Jan-2013' ;

refdn = datenum(refdate) ;

disp(['Select Philips temperature file, e.g. G:\Site\monitor_System_TempExamRoom.dat'])
[temp, dnum] = temperature_Philips_read;

figure
plot(dnum-refdn, temp,'r')
hold on
grid
title('Philips - red, logger - blue')
xlabel(['Days since ',refdate])
ylabel(['Temperature'])

lhumid = [0 0 0] ; % dummy
while length(lhumid) > 0
    disp(['Select humidity logging file'])
    [lhumid, ldnum] = humidity_logger_read ;
    
    plot(ldnum-refdn,lhumid,'b')
end



