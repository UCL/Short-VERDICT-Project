function A = xmlrecread(dinfo)
% xmlrecread(dinfo)  Read a single slice
%
% Does not check slice order, 
%
% See also PARREAD RECREAD PARREADALL

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
