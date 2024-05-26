function [data, dnum] = monitor_Philips_read(filenm)
% MONITOR_PHILIPS_READ  Reads monitoring data from Philips logs
%
%  [data, dnum] = monitor_Philips_read(filenm)
%  Files on scanner are located:
%    R3  G:\Site\monitor_XXXX.dat
%    R5  G:\monitoring\monitor_System_TempExamRoom.DAT
%
% Excludes out of hours measurements.
%
% Example use:
%  [data, dnum] = monitor_Philips_read ;  % Philips logs

%  plot(dnum,data,'r.')
%
% David Atkinson  D.Atkinson@ucl.ac.uk  November 2012
% See also humidity_logger_read  humidity_Philips_read

day_start = 8/24 ; % 8am
day_end = 19/24 ; % 7pm

prefnm = 'monitor_Philips' ;

if nargin < 1 || ~exist(filenm,'file')
    if ispref(prefnm,'filenm')
        filenm = getpref(prefnm,'filenm');
    else
        filenm = [] ;
    end
end

[fn,pn,FilterIndex] = uigetfile({'*.dat','Philips monitor .dat'},'Select .dat filename',filenm) ;

if fn == 0
    warning(['No dat file specified'])
    return
end

filenm = fullfile(pn,fn) ;

setpref(prefnm,'filenm',filenm) ;

[fid, message] = fopen(filenm) ;
if fid < 0
    disp([message])
end

fmt = '%19s  %f';
HL = textscan(fid,'%s',7,'Delimiter','') ;
C = textscan(fid,fmt,'Delimiter','') ;
fclose(fid) ;

str = HL{1}{7};
label = str(24:end) ;

dytm = C{1,1};  

data = C{1,2} ;
dnum = datenum(dytm,'yyyy-mm-dd HH:MM:SS') ;

% Remove sensor dud measures
loc = find(data==0 | data==123.8) ;

data(loc) = [] ;
dnum(loc) = [] ;
disp(['Removed ',num2str(length(loc)),' zeros or 123.8s'])

% Only report hours in the working day
floor_time = dnum - floor(dnum) ;
loc_outhours = find(floor_time < day_start | floor_time > day_end) ;
data(loc_outhours) = [] ;
dnum(loc_outhours) = [] ;
disp(['Removed ',num2str(length(loc_outhours)),' out of hours measures'])


if nargout==0
    refdate = '31-Jul-2015' ;
    refdn = datenum(refdate) ;
    plot(dnum-refdn,data,'r')
    xlabel(['Days since ',refdate])
    ylabel(['Value'])
    title([fn,': ',label])
end







