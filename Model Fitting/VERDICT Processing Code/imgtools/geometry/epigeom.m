fns = dselect ;

for ifn = 1:length(fns)
    dinfo = datparse(fns{ifn}) ;
    
    [v,m] = d2mat(dinfo,{'slice'},'op','fp') ;
    
    txt{ifn} = dinfo.ProtocolName ;
    IPP{ifn} = m.geom(5).IPP;
end

for ifn = 1:length(fns)
    disp([txt{ifn},' IPP ',num2str(IPP{ifn}(1)),'  ', num2str(IPP{ifn}(2))]) 
end

dinfo1 = datparse(fns{1}) ;
[v1,m1] = d2mat(dinfo1,{'slice'},'op','fp') ;
for ifn = 2:length(fns)
    dinfo = datparse(fns{ifn}) ;
    
    [v,m] = d2mat(dinfo,{'slice'},'op','fp') ;
    
    eshow(v-v1,'Name',[txt{ifn},' - ',txt{1}]) 
end



