function ims = individmeas(yout, opt)
% INDIVIDMEAS Recons of individual measures
%
% Loop through R's
%  make R new sub problems by extracting k-space, coordinates etc from
%  full data.
%  This is a bit messy but preserves sysmatv (which would otherwise
%  need further indexing

for iR = 1: length(opt.Rreq)
    optthis = opt ;
    optthis.verbose = false ;
    optthis.Rs2use = iR ;
    
    ythis = sysmatv(yout, 'selectyout', optthis) ;
    
    fmv = @(x,flag)sysmatv(x, flag, optthis) ;
    
    [xR, flag,relres,iter,resvec] = lsqr(fmv, ythis, [], opt.maxit) ;
    
    ims(:,iR) = xR ;
end

end % 