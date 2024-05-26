function [humid, dnum] = humidity_Philips_read(filenm)
% HUMIDITY_PHILIPS_READ  Reads humidity data from Philips logs
%
%  [humid, dnum] = humidity_Philips_read(filenm)
%  File on scanner is  G:\Site\monitor_System_HumExamRoom.dat
%
% Example use:
%  [humid, dnum] = humidity_Philips_read ;  % Philips logs
%  [lhumid, ldnum] = humidity_logger_read ;  % logger
%  plot(dnum,humid,'r.',ldnum,lhumid)
%
% David Atkinson  D.Atkinson@ucl.ac.uk  November 2012
% See also humidity_logger_read


prefnm = 'humidity_Philips' ;

if nargin < 1 || ~exist(filenm,'file')
    if ispref(prefnm,'filenm')
        filenm = getpref(prefnm,'filenm');
    else
        filenm = [] ;
    end
end

[fn,pn,FilterIndex] = uigetfile({'*.dat','Philips humidity .dat'},'Select .dat filename',filenm) ;

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

humid = C{1,2} ;
dnum = datenum(dytm,'yyyy-mm-dd HH:MM:SS') ;

loc = find(humid==0) ;

humid(loc) = [] ;
dnum(loc) = [] ;







