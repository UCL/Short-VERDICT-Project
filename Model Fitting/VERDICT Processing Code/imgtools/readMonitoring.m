function readMonitoring(opts)
% readMonitoring  Reads Philips monitoring data (humidity)
%
% Removes any entries with a value 0. Plots data in figure. Use figure 
% zoom or tooltip to examine the data points. 
% Note many points are acquired out of working hours.
%
% Example
% Copy to a local location the scanner file, e.g.:  
%   G:\monitoring\monitor_System_HumExamRoom.dat
%
%  readMonitoring                      use GUI to select the file manually
%
%  readMonitoring('fname',filename)    specify file (including path)
%
% David Atkinson, 2022.
%

arguments
    opts.fname {mustBeText}  % full file name 
end

prefnm = 'readMonitoring' ;

% Call uigetfile if valid file not specified
if ~isfield(opts,'fname') || isempty(opts.fname) || ~exist(opts.fname,'file')
    if ispref(prefnm,'deffilenm')
        deffilenm = getpref(prefnm,'deffilenm') ;
    else
        deffilenm = [] ;
    end

    disp('Select Philips monitoring .dat file')

    [fn,pn] = uigetfile({'*.dat','Philips monitoring .dat'},'Select .dat filename',deffilenm) ;
    if fn == 0
        warning('No .dat file specified')
        return
    end

    opts.fname = fullfile(pn,fn) ;

    setpref(prefnm,'deffilenm',opts.fname) 
end

% Open the file for reading
[fid, messg] = fopen(opts.fname,'r') ;
if fid<0
    disp(messg)
end

% Set fnext to filename including extension (without full path)
[~,fn,ext]=fileparts(opts.fname) ;
fnext = [fn,ext];
titlestr = fnext ; % default title

% Assumed format of file is
%   Description : Value
%    repeated above
%   Blank Line
%   Column Header Line
%   Data Lines

fmt = '%s' ;

% Header lines Description : Value
Cline = textscan(fid,fmt,1,'Delimiter','\r\n') ;
while ~isempty(Cline{1}{1})
    tline = Cline{1}{1} ;
    disp(tline)
    cp = textscan(tline,'%s',2,'Delimiter',':') ;
    if contains(cp{1}{1},'Hospital name') % use Hospital name for figure title
        titlestr = cp{1}{2} ;
    end
    
    Cline = textscan(fid,fmt,1,'Delimiter','\r\n') ;
end

% Column Header Line
% Read the header for the 3rd column
Cline = textscan(fid,fmt,1,'Delimiter','\r\n') ;
tline = Cline{1}{1} ;
cp = textscan(tline,'Date(YYYY-MM-DD) Time %s','Delimiter','') ;
disp(' ')
ylabelstr = cp{1}{1} ;
% remove this text if present:
k = strfind(ylabelstr,'(0=sensor not connected or broken)') ;
if ~isempty(k)
    ylabelstr = ylabelstr(1:k-1) ; ...
end
disp(ylabelstr)


% The main data
fmt = '%19s  %f';
C = textscan(fid,fmt,'Delimiter','') ;

fclose(fid) ;

dtimestr = C{1,1} ;
humid    = C{1,2} ;

dtime = datetime(dtimestr,'InputFormat','yyyy-MM-dd HH:mm:ss') ;

loc = find(humid==0) ; % Remove null entries
humid(loc) = [] ;
dtime(loc) = [] ;

figure('Name',fnext)
plot(dtime,humid)
grid on
title(titlestr)
xlabel('Date')
ylabel(ylabelstr)




