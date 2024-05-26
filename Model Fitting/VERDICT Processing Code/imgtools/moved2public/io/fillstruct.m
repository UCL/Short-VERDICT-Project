function struc = fillstruct(struc, field, val)
% FILLSTRUCT Fills fields of a structure
%
% struc = fillstruct(struc, field, val)
%
% See also D2MAT
%
for istr = 1:length(struc) 
    if isempty(getfield(struc(istr), field))
       struc(istr) = setfield(struc(istr), field, val) ;
    end
end
