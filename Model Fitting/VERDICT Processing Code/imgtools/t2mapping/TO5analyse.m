function TO5analyse
% TO5analyse PLots signal decay for multi-echo in user defined ROI

fns = dselect ;

nfn = length(fns) ;

dinfo = datparse(fns{1}) ;
[vto5{1}, mto5{1}] = d2mat(dinfo,{'slice','echo','itype'},'itype',11 ,'op','fp') ;
[vt2{1}, mt2{1}] = d2mat(dinfo,{'slice','itype'},'itype',7 ,'op','dv') ;

figure
imshow(vto5{1}(:,:,1,1),[],'Border','tight') % XData [0 to nx*dx]

roi = drawcircle('Color','r');
wait(roi)
bw = createMask(roi) ;
    
for ifn = 1:nfn
    dinfo = datparse(fns{ifn}) ;

    [vto5{ifn}, mto5{ifn}] = d2mat(dinfo,{'slice','echo','itype'},'itype',11 ,'op','fp') ;
    [vt2{ifn}, mt2{ifn}] = d2mat(dinfo,{'slice','itype'},'itype',7 ,'op','dv') ;
    
    necho = size(vto5{ifn},4) ;

    mean_sig = zeros([1 necho]) ;
  
    neg = zeros([1 necho]) ;
    pos = zeros([1 necho]) ;

    for iecho = 1: necho
        vals = vto5{ifn}(:,:,1,iecho) ;
        vals = vals(bw) ;
        mean_sig(1, iecho) = mean(vals(:)) ;
        neg(1,iecho) = mean_sig(1,iecho) - min(vals(:)) ;
        pos(1,iecho) = max(vals(:)) - mean_sig(1,iecho) ;
    end
    
    valst2 = vt2{ifn}(bw) ;
    
    errorbar([1:necho],mean_sig, neg, pos,'DisplayName',...
        [dinfo(1).ProtocolName,' T2:',num2str(mean(valst2))])
    
    hold on
    legend
    
end
lim = axis ;
lim(3) = 0 ;
axis(lim)
    
    