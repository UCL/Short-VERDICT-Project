function A = xmlparrecread(dinfo)
% xmlparrecread(dinfo)  Read a single slice from XML or PAR format rec file
%
% Does not check slice order, 
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also RECREAD PARREADALL D2MAT XMLPARSE PARPARSE

lab = dinfo ;

[fid, message] = fopen(lab.RecFileName,'rt') ;
if fid < 2
  error(message)
end


fseek(fid,0,1) ; % eof
len = ftell(fid) ;
fseek(fid,0,-1) ; % bof

if len ~= lab.RecFileSize
    error(['REC file has size ',num2str(len),'. Expected ',num2str(lab.RecFileSize)])
end


fseek(fid,lab.RecOffsetBytes,-1) ;
sizeA = [lab.Height lab.Width] ;
    
A = fread(fid,sizeA, 'uint16') ;
fclose(fid) ;

A = A.' ; % transpose to cope with MATLAB's column ordering 
end
