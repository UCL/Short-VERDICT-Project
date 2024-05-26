function [humid, dnum] = humidity_logger_read(filenm)
% HUMIDITY_LOGGER_READ Reads humidity from logging device files
%
%  [humid, dnum] = humidity_logger_read(filenm)
%  
% Example use:
%  [humid, dnum] = humidity_Philips_read ; % Philips logs
%  [lhumid, ldnum] = humidity_logger_read ;  % logger
%  plot(dnum,humid,'r.',ldnum,lhumid)
%
% David Atkinson  D.Atkinson@ucl.ac.uk  November 2012
% See also humidity_Philips_read


prefnm = 'humidity_logger' ;

if nargin < 1 || ~exist(filenm,'file')
    if ispref(prefnm,'filenm')
        filenm = getpref(prefnm,'filenm');
    else
        filenm = [] ;
    end
end

[fn,pn,FilterIndex] = uigetfile({'*.txt','Logger humidity .txt'},'Select .txt filename',filenm) ;

if fn == 0
    warning(['No txt file specified'])
    humid = [] ; dnum = [] ;
    return
end

filenm = fullfile(pn,fn) ;

setpref(prefnm,'filenm',filenm) ;

[fid, message] = fopen(filenm) ;
if fid < 0
    disp([message])
end

fmt = '%d%s%f%f%f';
C = textscan(fid,fmt,'Delimiter',',','HeaderLines',3) ;

fclose(fid) ;

dytm = C{1,2}; temp = C{1,3}; 

humid = C{1,4} ;
dnum = datenum(dytm,'dd/mm/yyyy HH:MM:SS');







