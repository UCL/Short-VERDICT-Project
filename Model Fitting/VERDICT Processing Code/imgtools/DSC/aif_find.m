function aif_find(vd)

if ndims(vd) ==2 && size(vd,2) == 1
    vd = vd' ;
end

sz = size(vd) ;
dimd = ndims(vd) ;
ndyn = size(vd,dimd) ;

vdmax = max(vd,[],dimd);
vdmin = min(vd,[],dimd);

vddiff = vdmax-vdmin ;
eshow(vddiff)

vddiffc = vddiff(:) ;
[~,locs] = sort(vddiffc,'descend') ;

vdr = reshape(vd,[prod(sz(1:dimd-1)) ndyn]) ;
top = vdr(locs(5:15),:) ;
aif_mean = mean(top,1) ;
figure
plot(aif_mean)

