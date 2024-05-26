dinfo = datparse(dselect) ;

useries = unique([dinfo.SeriesNumber]) ;
nf = length(useries) ;

%if nf > 1; error(['nf must be 1 at present']) ; end

% m=ceil(sqrt(nf)) ;
m=1;
n=ceil(nf/m) ;

figure
for jf = 1:nf
    [vde, mde, locde] = d2mat(dinfo,{'slice','dyn','echo','series'}, ...
        'series',useries(jf),'echo',1,'op','fp') ;
    
    tsl5 = squeeze(vde(:,49,5,:)) ;
    subplot(m,n,jf)
    imshow(tsl5,[])
    title(num2str(useries(jf)))
    drawnow
    

   % eshow(vde(:,:,5,:),mde.vdims(1:3),[num2str(useries(jf)),' ',dinfo(1).ProtocolName])
    
end




