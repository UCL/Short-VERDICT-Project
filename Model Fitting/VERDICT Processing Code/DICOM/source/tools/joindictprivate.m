function [dictJoined, dictOrig] = joindictprivate
% JOINDICTPRIVATE Joins MATLAB DICOM dictionary and a Private dictionary 
%
% dictJoined = joindictprivate
% [dictJoined, dictOrig] = joindictprivate
%
%  dictPrivate is Philips-R32-dict.txt
%
% Example
%   dictJoined = joindictprivate ;
%   dicomdict("set", dictJoined)
%    ...
%   dicomdict("factory") 
%
%
%   [dictJoined, dictOrig] = joindictprivate ;
%   dicomdict("set", dictJoined)
%    ...
%   dicomdict("set",dictOrig)
%
% David Atkinson, University College London
%
% Requires Image Processing Toolbox
%
% See also DICOMDICT
%

% Using fopen forces app packaging to detect and include this file
[fid, errmsg] = fopen('Philips-R32-dict.txt') ;
if fid < 0 
    error(errmsg)
end

dictPrivate = fopen(fid) ; % Puts full path in dictPrivate
fclose(fid) ;

if ~exist(dictPrivate,'file')
    error(['Dictionary file does not exist: ',dictPrivate])
end

dictOrig = dicomdict("get") ;  % original dictionary file on entry

dicomdict("factory")
dictFactory = dicomdict("get") ;

dictJoined = tempname ;

% Read in the factory dictionary and the private dictionary
fidFactory = fopen(dictFactory,'r') ;
fD = fread(fidFactory,'char') ;
fclose(fidFactory);

fidPrivate = fopen(dictPrivate,'r') ;
pD = fread(fidPrivate,'char') ;
fclose(fidPrivate) ;

fidJoined = fopen(dictJoined,'w') ;
%fseek(fidJoined,0,'eof') ;
fprintf(fidJoined,'%c',fD);
fprintf(fidJoined,'%c',pD);
fclose(fidJoined) ;

dicomdict("set",dictJoined) ;


dicomdict("set", dictOrig) ; % reset to original