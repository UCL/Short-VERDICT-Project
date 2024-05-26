function dinfo = dfastinfo(dcmfile, attrn, SOPClassUID)
% DFASTINFO Fast reading of some DICOM information
%   dinfo = dfastinfo(dcmfile, attrn, SOPClassUID)
%   dinfo = dfastinfo([], attrn)
%   
%   dinfo = dfastinfo([])  will read all attributes
%
%  SOPClassUID Cell array of SOPClassUIDs to be read, others will be
%  skipped. May also be 'all' (the default).
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% See also DSELECT
%

if nargin<2
    attrn = 'whole file' ; % tricks to reading whole file inc PixelData
end
if nargin < 3
    SOPClassUID = {'all'} ;
end

bitsalloc = 16 ; % reset later when read from file.
% Set DICOM Dictionary (see help on dicomdict for other options)
dict_in = dicomdict('get') ;
dicomdict('factory') ;
dict = dicomdict('get') ;
% This is usually a text file, need the corresponding mat file
dict(end-2:end) = 'mat' ;
dicomdict('set',dict_in) ; % Reset back
load(dict) % loads tags and values

if nargin < 1 || ~exist(dcmfile,'file')
   dcmfile = pref_uigetfile('dfastinfo', 'inputf') ;
end

[fid, message] = fopen(dcmfile, 'r', 'ieee-le') ; % little-endian header
if fid < 1
    warning(message) 
end

% Get file length
fseek(fid,0,'eof') ;
flen = ftell(fid) ;
fseek(fid,0,'bof') ;

if flen < 132
    warning(['File: ',dcmfile,' is not a DICOM file'])
    fclose(fid) ;
    dinfo = [] ;
    return
end

% read preamble and then next 4 characters
[preamble, count] = fread(fid, 128) ;  % 128 byte preamble
[dicmstr, count] = fread(fid, 4, 'uint8=>uint8') ;


if ~isequal(char(dicmstr)','DICM')
    disp(['Skipping fastread - no DICM in file: ',dcmfile ])
    fclose(fid);
    dinfo = [];
    return
end

dinfo.Filename = fopen(fid) ;
% First Group 0002 should be little endian explicit 

[group, element, VR, g2len ]  = dfastelemread(fid, 'explicit' ) ;
idx = tags(group+1,element+1) ;
nm = values(idx).Name ;
dinfo.(nm) = g2len ;

if group~=2 && element~=0
    warning(['Not Group 2, first element'])
end
while(ftell(fid) < 132+g2len)
    [group, element, VR, val ]  = dfastelemread(fid, 'explicit' ) ;
    idx = tags(group+1,element+1) ;
    nm = values(idx).Name ;
    dinfo.(nm) = val ;
end

switch dinfo.TransferSyntaxUID
    case '1.2.840.10008.1.2.1'  % Explicit VR Little Endian
        plicity = 'explicit' ;
    case '1.2.840.10008.1.2' % Implicit VR (Little?)endian
        plicity = 'implicit' ;
    case '1.2.840.10008.1.2.4.70' % JPEG loss less
        plicity = 'explicit' ; % ??
    otherwise
        warning(['TransferSyntax not implemented: ', dinfo.TransferSyntaxUID])
end

% Here we would close and reopen if endian change

% Keep reading until all attributes found or end of file
log_attr = ones(size(attrn)) ;
SOPClassUID_1st_seen = false; % SOPClassUID can appear again (!) in a private area.

while(ftell(fid) < flen-1 && sum(log_attr)>0)
    [group, element, VR, val ]  = dfastelemread(fid, plicity, tags, values, bitsalloc ) ;
    idx = tags(group+1,element+1) ;
    
    if idx == 0
        nm = ['Private_',dec2hex(group,4),'_',dec2hex(element,4)] ;
    else
        nm = values(idx).Name ;
    end
    dinfo.(nm) = val ;
    % disp([num2str(ftell(fid)),' (',dec2hex(group,4),',',dec2hex(element,4),') ',nm, ' : ', VR])
    
    % abort if incorrect SOPClassUID
    if ~isempty(SOPClassUID) && strcmp(nm,'SOPClassUID') && SOPClassUID_1st_seen==false
        loc_all = strcmp('all',SOPClassUID);
        if sum(loc_all)==0 % i.e. 'all' not in list
            scp = strcmp(dinfo.(nm), SOPClassUID) ;
            if sum(scp)==0
                dinfo = [] ;
                fclose(fid) ;
                return
            else
                SOPClassUID_1st_seen = true;
            end
        end
    end
        
    if strcmp(nm,'BitsAllocated')==1
        bitaslloc = val ;
    end
    
    loc = strcmp(nm,attrn) ;
    log_attr = log_attr - loc ; % log_attr has a one for every, as yet, 
                                % unfound attribute. Once all zeros, stops.
    
end

fclose(fid) ;




