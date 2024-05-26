function prostate2d
% PROSTATE2D
% Simulation of prostate distortion and noise.
%
%

n=168;  cols = 70:100 ;
ncol = length(cols) ;

img = prostate(n) ;

rng('default')

nR  = 2 ;
b0app = zeros(size(img));
b0mod = zeros(size(img));
imA = zeros([n ncol]);
ims = zeros([n ncol nR]) ; 
ims0b = zeros([n ncol nR]) ;
imj  = imA ;


for icol = 1:ncol
    col = cols(icol) ;
    
    xin = img(:,col) ; 
    
    % Filter (mild) to reduce ringing issues
    xin = imgaussfilt(xin) ;
    
    imA(:,icol) = xin ; % for output figure
    
    % Noise level to add later
    % 30 is bradly similar to b=1000 after averaging
    % 60 for b=0 after averaging
    
    snr = 60 ;
    signal = 0.7 ; % prostate phantom signal level
    mip = false ;
    
    opt.verbose = false ;
    opt.scale = 'balanced'; opt.vec = true ;
    opt.FOV = 220 ;
    opt.Rreq =  [1.2 1.6] ;  % undersampling factors requested
    opt.pFskip = [1 1] ;
    opt.tstep = [1 1] * 0.98e-3 ; % corresponding time steps (s)
    opt.partialFReq = 1 ;
    precond = false ; % seems to make solution a bit worse (as implemented).
    opt.precond = precond ;
    maxit = 10 ;
    opt.maxit = maxit ; % lsqr iterations
    
    % Coil sensitivities
    csens = ones([n 2]) ;
    csens(:,1) = exp(-([1:n]'- 0).^2/n/30) ;
    csens(:,2) = exp(-([1:n]'- n).^2/n/30) ;
    opt.csens = csens ;
    
    % motion induced phases
    if mip
        mp(:,1) = exp(2i*pi * (1-exp(-([1:n]-20).^2/n/60))) ;
        mp(:,2) = exp(2i*pi * exp(-([1:n]-100).^2/n/70)) ;
        % mp(:,3) = exp(2i*pi * (1-exp(-([1:n]-70).^2/n/60))) ;
        % mp(:,4) = exp(2i*pi * exp(-([1:n]-40).^2/n/70)) ;
    else
        mp = ones([n 2]) ;
    end
    opt.mp = mp ;
    
    
    % The B0 shows a dip near the prostate/rectum edge
    % b0fun takes a spatial postion as input
    % Dip amplitude varies with positon
    inrect = find(abs(xin-0.05)< 0.001) ;
    if ~isempty(inrect)
        irectA = inrect(1);
        irectP = inrect(end);
    else
        irectA = 0 ; irectP = 0 ;
    end
    
    low_signal_loc = find(xin < 0.06) ;
    %low_signal_loc = [] ;
    
    % opt.b0fun       = @(r) 0
    opt.b0fun       = @(ir)  b0rectum(ir,'irectA',irectA, 'irectP', irectP) ;
    opt.b0funmodify = @(ir)  b0rectum(ir, 'offset',5,'shift',1, ...
                            'low_signal_loc', low_signal_loc, ...
                            'irectA', irectA, 'irectP', irectP) ;
    
    % opt.b0funmodify = opt.b0fun ;
    
    [~,opt] = sysmatv(xin,'parse', opt) ;  % parse to set other opt fields
    
    % set time function. Takes itraj (unused) and index of k-space coord as
    % inputs. This index is converted to time by ckt (set in sysmatv)
    ckt = opt.ckt ;
    opt.Tfun  = @(itraj, ik) (ckt(ik))  ;
    
    % calc k-space yout
    [yout, optout] = sysmatv(xin,'notransp', opt) ;
    
    % Add noise. Use noise variance determined from image signal and expected
    % SNR. Although yout may contain multiple measures, the noise added should
    % be correct.
    
    yout = addnoise(yout,'snr',snr,'signal',signal,'indomain','kspace','scale','balanced') ;
    
    % recon
    opt.b0fun = opt.b0funmodify ;
    
    if mip
        % Estimate motion induced phase (and f0 drift)
        newmp = mpestimate(yout, opt) ;
        opt.mp = newmp ;
    end
    
    % Get recons of individual measures with current B0 estimate
    ims_this = individmeas(yout, opt) ;
    ims(:,icol,1) = ims_this(:,1) ;
    ims(:,icol,2) = ims_this(:,2) ;
    
    % Get recons of individual measures with 0 B0 estimate
    opt0b = opt;
    opt0b.b0fun = @(r) 0 ;
    ims_this = individmeas(yout, opt0b) ;
    ims0b(:,icol,1) = ims_this(:,1) ;
    ims0b(:,icol,2) = ims_this(:,2) ;
    
    
    % Do LSQR on k-space
    opt.verbose = false ;
    fmv = @(x,flag)sysmatv(x, flag, opt) ;
    
    [b0app, b0mod] = b0image(b0app, b0mod, col, optout) ;
    
    [x1, flag,relres,iter,resvec] = lsqr(fmv, yout, [], maxit) ;
    
    imj(:,icol) = x1 ;
end

eshow([imA ims0b(:,:,1) ims0b(:,:,2) ims(:,:,1) ims(:,:,2) imj], 'Name', ...
    ['R ',num2str(opt.Rreq),'.  tstep: ',num2str(opt.tstep*1000), ...
    ' pF: ',num2str(opt.partialFAct)])
eshow([b0app b0mod])

end



