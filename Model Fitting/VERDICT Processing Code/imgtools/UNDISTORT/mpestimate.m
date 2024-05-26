function newmp = mpestimate(yout, opt)
% MPESTIMATE Estimate motion induced phase (and f0 drift?)
%
% Loop through R's
%  make R new sub problems by extracting k-space, coordinates etc from
%  full data.
%  This is a bit messy but preserves sysmatv (which would otherwise
%  need further indexing

if opt.verbose, figure('Name','angle xR'), end
for iR = 1: length(opt.Rreq)
    optthis = opt ;
    optthis.verbose = false ;
    optthis.Rs2use = iR ;
    
    optthis = rmfield(optthis, 'mp') ;
    ythis = sysmatv(yout, 'selectyout', optthis) ;
    
    fmv = @(x,flag)sysmatv(x, flag, optthis) ;
    mfun = @(x, flag)msysmatv(x, flag, optthis) ;
    if opt.precond
        [xR, flag,relres,iter,resvec] = lsqr(fmv, ythis, [], opt.maxit, mfun) ;
    else
        [xR, flag,relres,iter,resvec] = lsqr(fmv, ythis, [], opt.maxit) ;
    end
    
    if opt.verbose
        plot(abs(xR)), hold on
        yyaxis right
        plot(angle(xR)), hold on
        plot(movmean(angle(xR),5))
        plot(movstd(angle(xR),5))
        yyaxis left
    end
    newmp(:,iR) = exp(1i*angle(xR)) ;
end

end % mpestimate