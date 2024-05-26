function [group, element, VRs, val ] = ...
    dfastelemread(fid, plicity, tags, values, bitsalloc)
% DFASTELEMREAD Read a DICOM data element
%
% [group, element, VRs, val ] = dfastelemread(fid, plicity, tags, values, bitsalloc)
%
% fid - file identifier. File opened in calling function with correct
%       endedness
% plicity - 'implicit' or 'explicit' (for Value Representation)
% tags, values - from data dictionary
% bitsalloc - bits allocated (allows reading of old ACR/NEMA files)
%             defaults to 16
%
% group - data element group number (decimal)
% element - data element element number (decimal)
% VRs     - Value Representation (string). Taken from data if explicit VR,
%           from data dictionary if implicit VR
% val     - the data.
%
% Function leaves file pointer ready to read the next data element
% group number.
%
% Function is called recursively for SQ elements. Will read dicomdir.
% but leaves some entries in val - needs further decoding.
%
% No longer reports minor standard violations (messages annoying)
%
% David Atkinson, 
% Radiological Sciences, Guy's Hospital, London.
% University College London
%
% See also DICOM

if nargin < 4
    bitsalloc = 16 ;
end

group = fread(fid, 1, 'uint16') ;
element = fread(fid, 1, 'uint16') ;

if group==hex2dec('fffe') 
    if element==hex2dec('e0dd') || element==hex2dec('e00d') || element==hex2dec('e000')
        % Item or Sequence Delimitation or Item
        val = ''; VRs=[] ;
        dummy = fread(fid,1,'uint32') ;
        return
    end
end
    
switch plicity
    case 'explicit'
        VR = fread(fid,2,'uchar') ;
        VRs = char(VR') ;
        
        switch VRs
            case { 'OB', 'OW', 'SQ', 'UN' }
                res = fread(fid, 2,'uchar') ; % reserved
                
                len = fread(fid, 1, 'uint32') ;
                
                
            otherwise
                len = fread(fid, 1, 'uint16') ;
        end
        
    case 'implicit'
        
        idx = tags(group+1, element+1) ;
        VRs = values(idx).VR ;
%         idic = elem2dic(dict, group, element) ;
%         if length(idic) == 0
%             VRs = 'LO' ;       % private groups from .SPI or ACR/NEMA files
%         else
%             VRs = char(dict.VR(idic)) ;
%         end
        
        len = fread(fid, 1, 'uint32') ;
        
    otherwise
        error([ 'Must be implicit or explicit.'])
end

% 4294967295  = hex2dec('FFFFFFFF'). recoded for speed
if len ==  4294967295 & strcmp(VRs,'SQ')==0
    VRs
    values(idx).Name
    warning([ ' ''Undefined length'' not supported, except for VR of SQ'])
end

% read the item(s)

switch VRs
    case 'AE'
        if len > 16, warning([ 'Application entity max 16 bytes.']), end
        val = fread(fid, len, 'uchar') ;
    case 'AS' % age string
        if len ~= 4, warning([ 'Age string should have four characters.']), end
        val = fread(fid, len, 'uchar') ;
    case 'AT' % attribute tag
        if len ~= 4, warning([ 'Attribute tag should have four characters.']), end
        val = fread(fid, len/2, 'uint16') ;
    case 'CS' % code string
        % if len > 16, warning([ 'Code string max 16 bytes.']), end
        val = fread(fid, len, 'uchar') ;
    case 'DA' % date
        val = fread(fid, len, 'uchar') ;
    case 'DS' % decimal string
        % if len > 16, warning([ 'Decimal string max 16 bytes.']), end
        val = fread(fid, len, 'uchar') ;
    case 'DT' % date time
        if len > 26, warning([ 'Date time max 26 bytes.']), end
        val = fread(fid, len, 'uchar') ;
    case 'FL' % 32 bit fp number
        %if len ~= 4, warning([ 'Floating single expects 4 bytes, requested ', ...
        %        num2str(len)]), end
        val = fread(fid, len/4, 'float32') ;
    case 'FD' % 64 floating point
        % if len ~= 8, warning([ 'Floating double expects 8 bytes']), end
        val = fread(fid,len/8, 'float64') ;
    case 'IS' % integer string
        %if len > 12, warning([ 'IS max 12 bytes.']), end
        val = fread(fid, len, 'uchar') ;
    case 'LO' % long string
        val = fread(fid, len, 'uchar') ;
    case 'LT' % long text
        val = fread(fid, len, 'uchar') ;
    case 'NONE' % used in reading SQ items
        val = fread(fid, len, 'uchar') ;
    case 'OB' % other byte string
        val = fread(fid, len, 'uint8') ; %????
    case 'OW' % other word string
        if bitsalloc== 16
            val = fread(fid, len/2, 'uint16') ;
        elseif bitsalloc == 12
            val = fread(fid, len/2*16/12, 'ubit12=>uint16') ;
        elseif bitsalloc == 8
            val = fread(fid, len, 'uint8') ;
        else
            warning([' unsupported bits allocated: ', num2str(bitsalloc)])
        end
        
    case 'OW/OB'
        warning([ 'VR OW/OB guessing OW' ])
        if bitsalloc== 16
            val = fread(fid, len/2, 'uint16') ;
        elseif bitsalloc == 12
            val = fread(fid, len/2*16/12, 'ubit12=>uint16') ;
        elseif bitsalloc == 8
            val = fread(fid, len, 'uint8') ;
        else
            warning([' unsupported bits allocated: ', num2str(bitsalloc)])
        end
    case 'PN' % person name
        val = fread(fid, len, 'uchar') ;
    case 'SH' % short string
        val = fread(fid, len, 'uchar') ;
    case 'SL' % signed long
        %if len ~= 4, warning([ 'Signed long expects 4 bytes.']), end
        val = fread(fid, len/4, 'int32') ;
    case 'SQ' % sequence of items
        sqlen = len ;
        ival = 0 ;
        if sqlen ~= hex2dec('FFFFFFFF')
            fposeos = ftell(fid) + sqlen ;
            while ftell(fid) < fposeos
                % [itgp, itelem, itVRs, itval] = d3elemread(fid,'implicit', dict) ;
                [itgp, itelem, itVRs, itval] = dfastelemread(fid,'explicit') ;
                ival = ival + 1 ;
                val(ival).item = itval ;
            end
        else
            % udefined length for sequence
            % Try to jump not wasting time on reading.
            
            % search for Sequence Delimitation Item (FFFE,E0DD)
            poscurr = ftell(fid) ;
            itgp = group ; itelem = element  ;
            while itgp~=hex2dec('fffe') && itelem~=hex2dec('e0dd')
                fseek(fid,poscurr-6,'bof') ;
                itgp = fread(fid, 1, 'uint16') ;
                itelem = fread(fid, 1, 'uint16') ;
                testv = fread(fid, 1, 'uint32') ;
                poscurr = ftell(fid) ;
            end
            val = 'undefined length item(s)' ;
            
%             itgp = group ; itelem = element ;
%             while itgp~=hex2dec('fffe') & itelem~=hex2dec('e0dd')
%                 % [itgp, itelem, itVRs, itval] = d3elemread(fid,'implicit', dict) ;
%                 [itgp, itelem, itVRs, itval] = dfastelemread(fid,'implicit', tags, values) ;
%                 ival = ival + 1 ;
%                 val(ival).item = itval ;
%             end
        end
        
    case 'SS' % signed short
        % if len ~= 2, warning([ 'signed short expects 2 bytes.' ]), end
        val = fread(fid, len/2, 'int16') ;
    case 'ST' % short text
        val = fread(fid, len, 'uchar') ;
    case 'TM' % time
        if len >16 , warning([ 'time max 16 bytes.' ]), end
        val = fread(fid, len, 'uchar') ;
    case 'UI' % unique identifier
        if len >64 , warning([ 'ui max 64 bytes.' ]), end
        val = fread(fid, len, 'uchar') ;
    case 'UL' % unsigned long
        % if len ~= 4, warning([ 'Unsigned long expects 4 bytes.']), end
        val = fread(fid, len/4, 'uint32') ;
    case 'UN' % unknown
       % !!! Implememnt check for unknown length
        val = fread(fid, len, 'uchar') ;
    case 'US' % unsigned short
        % if len ~= 2, warning([ 'Unsigned short expects 2 bytes.']), end
        val = fread(fid, len/2, 'uint16') ;
    case 'US or SS'
        val = fread(fid, len/2, 'uint16') ;
    case 'UT' % unlimited text
        val = fread(fid, len, 'uchar') ;
    otherwise
        warning([ 'Unrecognised VR: ',VRs])
        val = [] ;
end

switch VRs
    case { 'AT', 'FL', 'FD', 'OB', 'OW', 'SL', 'SQ', 'SS', 'UL', 'US' }
    otherwise
        val = char(val') ;
        val = deblank(val) ;
end

