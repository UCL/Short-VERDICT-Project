

dinfo = datparse(dselect) ;

[v,m] = d2mat(dinfo,{'slice','dyn'},'op','fp') ;

ffn = dinfo(1).Filename;
[pn, fn, ext] = fileparts(ffn) ;

figure('Name',[fn, ext])
plot(v(122,:,13,1))
hold on
plot(10*v(122,:,13,end))



