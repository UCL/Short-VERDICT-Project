function rnorm = simul(s)
%
% simul(simulopt)
%
% Further tests;
%  write k2ip and i2kp with corresponding direction and scaling to real
%  world. Use these as validation for E and EH (EH with no coils as FFT is
%  unitary).
%
%

dt_s = s.dt_s ;
nfe = s.nfe ;
npe_u = s.npe_u ; % number phase encodes (SENSE undersampling)
npe_f = s.npe_f ; % number phase encodes (
ntraj = s.ntraj ;  % 2 for up and down

partialF = s.partialF ;

hf = figure('Name',[s.name,'  ',datestr(datetime('now'))]) ;
sm = 3 ; sn = 3 ; % subplot size

img = prostate(nfe) ;
img = imresize(img,[nfe npe_f]) ;

R = npe_f/npe_u ;
disp(['SENSE undersampling factor R: ',num2str(R), ...
      ' Partial Fourier: ',num2str(partialF)])


sens_lower_thresh = s.sens_lower_thresh ; % Lower threshold for coil SOS sensitivities, all values floored here for pre-conditioning

tol = s.tol; % lsqr
maxit = s.maxit ;


% Assume coils in FE planes at edge of image
ncoil = s.ncoil ;
cph_mult = s.cph_mult ; % degrees for coil phase multiplier (use 0 for non) large for testing

if ncoil == 1
    csens = ones([nfe npe_f]) ;
elseif ncoil==4
    ccent = {[round(nfe/3) -5], [round(nfe*2/3) -5], [round(nfe/3) npe_f+6], [round(nfe*2/3) npe_f+6]} ;
    csens = zeros([nfe npe_f ncoil]) ;
    a = round(nfe/5) ; % coil radius
    
    for jcoil=1:ncoil
        cphase = (jcoil-1)/ncoil * cph_mult ;
        cphase = exp(1i*cphase/360*2*pi) ;
        
        for jife = 1: nfe
            for jipe = 1:npe_f
                Rc = jife-ccent{jcoil}(1) ; % distance to coil radius
                Z = jipe-ccent{jcoil}(2) ; % distance to coil plane
                csens(jife, jipe, jcoil) = cphase * coil_sensitivity(a,Rc,Z) ;
            end
        end
    end
end
if s.csens_obj_only
   % Coil sensitivity only known where there is object
   loc = find(img > max(img(:))/30) ;
   img_mask = zeros(size(img)) ;
   img_mask(loc) = 1 ;
   
   csens = csens .* repmat(img_mask,[1 1 ncoil]) ; 
end

subplot(sm, sn, 7)
montage(abs(csens).^0.6); title('csens')

csos = sqrt(sum(csens.*conj(csens),3)) ;
csos_max = max(csos(:)) ;
low = csos < csos_max*sens_lower_thresh ;
csos(low) = csos_max*sens_lower_thresh ;

pcsos = 1./csos ;


PS_pe = 1 ; PS_fe =1; % pixel spacing in mm

FOV_pe = PS_pe * npe_f ;
FOV_fe = PS_fe * nfe ;

xc = ([1:npe_f]-DCgp([1:npe_f],2)) * PS_pe ;
yc = ([1:nfe]-DCgp([1:nfe],2)) * PS_fe ;

XPE = repmat(xc,[nfe 1]) ;
YFE = repmat(yc',[1 npe_f]) ;

kpev = ([1:npe_u]-DCgp([1:npe_u],2)) * R / FOV_pe ;
kfev = ([1:nfe]-DCgp([1:nfe],2) )/ FOV_fe ;

KPE = repmat(kpev,[nfe 1]) ;
KFE = repmat(kfev',[1 npe_u]) ;


B0_Hz = s.B0_amp * exp(-(XPE.^2+YFE.^2)/nfe) ;
figure(hf), subplot(sm,sn,1), imagesc(B0_Hz), colorbar, title('B0 map')


% Partial F sampling
kmask = ones([nfe npe_u ntraj]) ;
pflim = npe_u - round(partialF*npe_u) ;
disp(['Actual partial Fourier: ',num2str(100*(npe_u-pflim)/npe_u),'%, ',num2str(npe_u-pflim),...
    ' out of ',num2str(npe_u),' lines.'])


T{1} = XPE/PS_pe * dt_s ;
T{2} = fliplr(T{1}) ;
kmask(:,1:pflim,1) = 0 ; 
if ntraj == 2
    kmask(:,(npe_u-pflim):npe_u,2) = 0 ;
end

if mod(npe_u,2)~=1, warning(['Cant use flip for trajectories above']), end
figure(hf), subplot(sm, sn, 2)
for jtraj = 1:ntraj
    plot(T{jtraj}(1,:))
    hold on ;
end
ylabel('Time'), xlabel('PE line'), title('Trajectory times')


% Scaling in CG.
%  FFT represents an integration and as such should be scaled with the 
%  integration variable. Often just use a 1/N on one of the directions.
%  This is correct, for example if pixel size is 1, then FOV is N and the
%  k-space step size is 1/N. Sometimes 1/sqrt(N) is used in both directions
%  and would correspond to a pixel size of 1/sqrt(N) and a FOV of sqrt(N).
%
% As integrations, K-space shoud be scaled by PS_pe * PS_fe  
% and image space by R / (FOV_pe * FOV_fe) / (ntraj * ncoil) 
%
% BUT, we are trying to represent a matrix A and A^H here so however we
% scale the elements of A, we need to do the same when computing A^H. Hence
% we set here one matrix-vector scale.

mv_scale = sqrt(npe_f * nfe * ntraj * ncoil) ;

% Compute b
b = afun(img(:), 'notransp') ;
disp(['Norm b (Ax=b) before noise: ',num2str(norm(b)) ])

% add noise in image domain (to get sensible scaling)
if s.addnoise
    % convert to image to add noise
    brs = reshape(b, [nfe, npe_u, ntraj, ncoil]) ;
    imbrs = k2isbs(brs) ;
    imbrs = addimnoise(imbrs, s.snr) ;
    b = i2ksbs(imbrs) ;
    clear imbrs  brs
    b = b(:) ;
end


disp(['Norm of b after noise added is: ', num2str(norm(b))])
disp(['Norm of img is: ',num2str(norm(img(:))) ])

img_check = afun(b(:), 'transp') ; % Conjugate Phase 
disp(['Norm of A^H b is: ',num2str(norm(img_check)) ])

% FFT is unitary so CP gives a k2i type transform
% Note that when there are coil sensitivities, E is not unitrary, i.e. E^H
% is not E^-1. In particular, there will be scaling differences 

subplot(sm, sn, 3)
imagesc(reshape(abs(img_check),[nfe npe_f])), colormap('gray'), axis square
title('img check (CP)')

ftb = i2ksbs(reshape(b,[nfe npe_u ntraj ncoil])) ;

subplot(sm, sn, 4)
srs = sqrt(sum((ftb.*conj(ftb)),4)) ;
imshow(srs(:,:,1))
title('SOS fft on b, traj 1')
if ntraj > 1
   subplot(sm, sn, 5), imshow(srs(:,:,2)), title('SOS fft on b, traj 2')
end

switch s.x0
    case 'empty'
        x0=[] ;
case 'img'
        x0 = img(:) ;
    otherwise
        error(['Unknown x0 option: ',s.x0])
end

[x1, flag,relres,iter,resvec] = lsqr(@afun, b, tol, maxit, @mfun,[],x0) ;

disp(['LSQR finished after ',num2str(iter), ...
    ' iterations with flag: ',num2str(flag), ...
    ', relres ',num2str(relres), ...
    ', norm(x1): ',num2str(norm(x1))  ])

figure(hf), subplot(sm, sn, 6),
plot(resvec)
xlabel('Iter'), ylabel('residual')


im_out = reshape(x1,[nfe npe_f]) ;

rnorm = norm(im_out-img)/norm(img) ;

disp(['Rel norm of final difference: ',num2str(rnorm)])

subplot(sm, sn, 8), imagesc(abs(img)), title('img')
subplot(sm, sn, 9), imagesc(abs(im_out)), title('im_out')
eshow(cat(2,im_out, (im_out - img)),'Name',[s.name,' im_out-img'])

    function y = afun(x, transp_flag)
        if strcmp(transp_flag,'notransp') % FWD model one image to k-spaces
            % S = A F C Im
            S = zeros([nfe, npe_u, ntraj, ncoil]) ;
            imgx = reshape(x,[nfe npe_f]) ;
            for icoil = 1:ncoil
                for itraj = 1:ntraj
                    for ikpe = 1:npe_u
                        for ikfe = 1:nfe
                            for iipe = 1:npe_f
                                for iife = 1:nfe
                                    S(ikfe,ikpe,itraj,icoil) = S(ikfe,ikpe, itraj, icoil) + ...
                                        csens(iife, iipe, icoil) * imgx(iife, iipe) * exp(2i*pi*(KPE(ikfe,ikpe)*XPE(iife,iipe) + ...
                                        KFE(ikfe,ikpe)*YFE(iife,iipe) + B0_Hz(iife, iipe)*T{itraj}(ikfe,ikpe))) ;
                                end
                            end
                        end
                    end
                end
            end
            
            S = S / mv_scale ;
            
            S = S .* kmask ; % kmask is 2D - compatible element-wise multiplication
            y = S(:) ;
            
        elseif strcmp(transp_flag,'transp') % A^H y     K-spaces to image
            
            W = zeros([nfe npe_f]) ;
            Sx = reshape(x,[nfe npe_u ntraj ncoil]) ; 
            Sx = Sx .* kmask ;
            for ikpe = 1:npe_u
                for ikfe = 1:nfe
                    for iipe = 1:npe_f
                        for iife = 1:nfe
                            for itraj = 1:ntraj
                                for icoil = 1:ncoil
                                    % Integral over k
                                    W(iife,iipe) = W(iife,iipe) + ...
                                        conj(csens(iife, iipe,icoil)) * Sx(ikfe, ikpe, itraj,icoil) * exp(-2i*pi*(KPE(ikfe,ikpe)*XPE(iife,iipe) + ...
                                        KFE(ikfe,ikpe)*YFE(iife,iipe) + B0_Hz(iife, iipe)*T{itraj}(ikfe,ikpe))) ;
                                end
                            end
                        end
                    end
                end
            end
            
            y = W(:) / mv_scale ;
        end
    end

    % pre-condition - certainly helps in simulation
    function y= mfun(x, transp_flag)
        if strcmp(transp_flag,'notransp') % 'notransp') returns M\x 
            % If FWD model A is m x n, then M should be n x n. So, n must
            % be image size and input x must be image
            W = reshape(x,[nfe npe_f]) ;
            cw = pcsos .* W ;
            
            y = cw(:) ;
            
        elseif strcmp(transp_flag,'transp') % returns M'\x.
            W = reshape(x,[nfe npe_f]) ;
            cw = conj(pcsos) .* W ; % transpose is of full diag matrix. pcsos is square.
            
            y = cw(:) ;
        end
    end


end

        
