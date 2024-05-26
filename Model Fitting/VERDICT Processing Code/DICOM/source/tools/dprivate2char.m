function dprivate2char(dinfo, d2)

fields = fieldnames(dinfo) ;
fields2 = fieldnames(d2) ;

fieldsu = union(fields, fields2) ;

for ifield = 1: length(fieldsu)
    if contains(fieldsu{ifield},'Private') && ~contains(fieldsu{ifield},'Creator')
        if isfield(dinfo,fieldsu(ifield))
            c1 = char(dinfo.(fieldsu{ifield}))' ;
        else
            c1 = '';
        end

        if isfield(d2,fieldsu(ifield))
            c2 = char(d2.(fieldsu{ifield}))' ;
        else
            c2 = '';
        end

        if ~strcmp(c1,c2)
            disp([fieldsu{ifield},': ',c1,' : ',c2])
        end
    end
end