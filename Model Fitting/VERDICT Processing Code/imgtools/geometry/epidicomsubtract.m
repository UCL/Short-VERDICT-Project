function epidicomsubtract

sl = 5 ;

fn36 = dselect('message','ref scan 36') ;

d36 = datparse(fn36) ;

fns = dselect('message','remaining EPI') ;

[v36,m36] = d2mat(d36, {'slice'},'op','fp') ;

img = v36(:,:,sl) ;
imgf = img-circshift(img,1) ;

for ifn = 1:length(fns) 
    dinfo = datparse(fns{ifn}) ;
    
    [v,m] = d2mat(dinfo,{'slice'},'op','fp') ;
    
    imgf = cat(2,imgf, v(:,:,sl)-img) ;
end
eshow(imgf)
