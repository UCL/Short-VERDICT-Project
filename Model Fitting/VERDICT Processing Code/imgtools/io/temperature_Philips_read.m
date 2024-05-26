function [temp, dnum] = temperature_Philips_read(filenm)
% TEMPERATURE_PHILIPS_READ  Reads temperature data from Philips logs
%
%  [temp, dnum] = temperature_Philips_read(filenm)
%  File on scanner is  G:\Site\monitor_System_TempExamRoom.dat
%
% Example use:
%  [temp, dnum] = humidity_Philips_read ;  % Philips logs

%  plot(dnum,temp,'r.')
%
% David Atkinson  D.Atkinson@ucl.ac.uk  November 2012
% See also humidity_logger_read humidity_Philips_read


prefnm = 'temperature_Philips' ;

if nargin < 1 || ~exist(filenm,'file')
    if ispref(prefnm,'filenm')
        filenm = getpref(prefnm,'filenm');
    else
        filenm = [] ;
    end
end

[fn,pn,FilterIndex] = uigetfile({'*.dat','Philips temperature .dat'},'Select .dat filename',filenm) ;

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
C = textscan(fid,fmt,'HeaderLines',10,'Delimiter','') ;

fclose(fid) ;

dytm = C{1,1};  

temp = C{1,2} ;
dnum = datenum(dytm,'yyyy-mm-dd HH:MM:SS') ;

loc = find(temp==0) ;

temp(loc) = [] ;
dnum(loc) = [] ;







