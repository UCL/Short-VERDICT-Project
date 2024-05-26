function rnorm = joint_simul(s)
% JOINT_SIMUL  SENSE, Partial Fourier and joint B0 and EPI reconstruction.
% Joint reconstruction follows thesis of Matakos.
%
% rnorm = joint_simul(simulopt)
% Usually run:
%  tests = runtests('simulTest.m') 
%
% When B0 is an unknown, it shoud be real. (Note in Matakos code, he uses 
%  which is i*B0) 
%
% A rigid image shift and B0 offset are degenerate for one direction of EPI
% only. 
% Seems to be interplay between image and B0. In bfun
% they are multiplied prior to Fourier Transform. 
%
% Pre-conditioning
%  Not certain about how to apply (Simon does not use). In earlier tests it
%  made an improvement but this needs repeating.
%
% TO investigate:
% Compare with Hansen's ismrm_non_cartesian_sense for preconditioning,
% pre-whitening and regularisation.
%
% Possibly add regularizers



%
% Image CG as before
% B0 CG uses linearized scheme
%  FWD model but using a difference k-space y'
%    y' = B B0new
%    y' = y - A C f  + B B0est
% where f is image vector, F is diag(f), A is usual system matrix using B0est and 
%  B is  - diag(t) A C F
%
% Solves for new B0 (not difference).
%
% Transpose operation is then
%  F^H C^H A^H diag(t)


% namestring for labelling outputs
namestr = [s.name,'  ',datestr(datetime('now')) ] ;
disp(' ')
disp(namestr)

dt_s = s.dt_s ;
nfe = s.nfe ;
npe_u = s.npe_u ; % number phase encodes (after SENSE undersampling, before 
                  % partial Fourier)
npe_f = s.npe_f ; % number phase encodes in full size i.e. recon image size
% npe_a will be actual number of phase encodes acquired

ntraj = s.ntraj ;  % 2 for both up and down

R = npe_f/npe_u ; % SENSE factor

% Partial F sampling
npe_a = round(s.partialF * npe_u) ;
disp(['SENSE undersampling factor R: ',num2str(R), ...
      ' Actual partial Fourier: ',num2str(100*npe_a/npe_u), '%'])
disp(['BW/pixel = ',num2str(1/(npe_u * dt_s)),' Hz. WFS = ', ...
    num2str(440*npe_u*dt_s),' pixels at 3T.'])


hf = figure('Name',['summary ',namestr]) ;
sm = 3 ; sn = 3 ; % subplot size

img = rot90(prostate(nfe),-1) ;
img = imgaussfilt(img,0.7) ; % filter to remove hard edges
img = imresize(img,[nfe npe_f]) ;

time_seg = s.time_seg ; % Boolean

sens_lower_thresh = s.sens_lower_thresh ; % Lower threshold for coil SOS 
           % sensitivities, all values floored here for pre-conditioning
pc_thresh = s.pc_thresh ;

tol = s.tol; % lsqr
maxit = s.maxit ;

nouter = s.nouter ;
if s.B0_joint == false && nouter > 1
    nouter = 1 ;
    disp(['Setting nouter to 1 as no joint B0 estimation.'])
end

% Assume coils in FE planes at edge of image
ncoil = s.ncoil ;
cph_mult = s.cph_mult ; % degrees for coil phase multiplier (use 0 for non) 
                        % large for testing

if ncoil == 1
    csens = ones([nfe npe_f]) ;
elseif ncoil==4
    ccent = {[round(nfe/3) -5], [round(nfe*2/3) -5], ...
               [round(nfe/3) npe_f+6], [round(nfe*2/3) npe_f+6]} ;
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
montage(abs(csens).^0.6); title('csens'), colorbar

csos = sqrt(sum(csens.*conj(csens),3)) ;
csos_max = max(csos(:)) ;
low = csos < csos_max*sens_lower_thresh ;
csost = csos ;
csost(low) = csos_max*sens_lower_thresh ;

pcsos = 1./csost ;


% B0 maps:
% B0_act   - actual (oracle) B0 map used to simulate acquired k-space b.
% B0_init  - B0 map used initally e.g. from B0 measurement
% B0_curr  - current best estimate of B0
% B0x      - the B0 passed into bfun by the LSQR code.

% img is [nfe npe_f], B0 should be the same
ri0 = ([1:nfe]-DCgp([1:nfe],2))' ; % row index, 0-centred
ci0 = ([1:npe_f]-DCgp([1:npe_f],2)) ;

B0_act = s.B0_amp * exp(-(ri0.^2+ci0.^2)/nfe) ; % uses implicit expansion
figure(hf), subplot(sm,sn,1), imagesc(B0_act), colorbar, title('B0\_act map')


% Sequence Timing and variables
% S is predicted k-space [nfe npe_a ntraj ncoil]
%
%  S 
%  1 2      22 indS         -15     0   6  up itraj=1
%  ---------> npe_a         <------------
%  |                                
%  |                           -6   0      15  down itraj=2
%  V nfe                       -------------->
%    
%  For npe_a = 22, npe_f = 31, partialF 0.7
%  indS  1  2  3  4  5 .. 22 
%  it2ts -6 -5 ... 15 ms
%  
%  itraj 1 "up" from right to left in this scheme
%  it2iS{1}  22 21 20 ... 1     time point to index into S
%  it2iS{2}  1   2  3 ... 22
%  
%  iS2ifull{1}  1 2 3 ... 22  index in S to full (npe_u) k-space i.e.
%  iS2ifull{2}  10 11 ... 31   accounts for SENSE and 'zero fills' for pF
%
%  time point to k-space coord (after partial Fourier, not inc FOV scaling)
%  it2ka{1}  6 5 ... -15  (if R>1, separation here would be greater than 1)
%  it2ks{2}  -6-5 ... 15
%
% Assumes there are npe_a time 'segments' 
%    90 D 180 D   epi+ epi- TE epi+ epi- epi+ epi- epi+
%       it2ts      -2   -1   0  1    2    3    4    5   * dt_s    Traj 1
%      time index (segment or line) to time in seconds
%

ntseg = npe_a ;
regk = ci0 ;
cui0 = ([1:npe_u]-DCgp([1:npe_u],2)) ;
uk =  cui0 * R ;
it2ka(:,1) = fliplr(uk(1:npe_a)) ;
it2ka(:,2) = uk(end-npe_a+1:end) ;

it2ts{1} = dt_s * cui0(end-npe_a+1:end) + s.es(1) ;
it2ts{2} = dt_s * cui0(end-npe_a+1:end) + s.es(2) ;

SMt = gen_sincmat(regk, uk ) ;  % Interp from full k-space to SENSE 
                                % undersampled (partial F taken care of by
                                % seleting approriate entries from matrix).
                                % Allows for non-integer SENSE

it2iS{1} =  [npe_a:-1:1] ;
it2iS{2} =  [1:npe_a] ; 

iS2ifull{1}=[1:npe_a] ;    % "full" is k-space with size npe_u  
iS2ifull{2}=[1:npe_a] + npe_u - npe_a ;

iS2ts{1} = fliplr( it2ts{1} ) ;
iS2ts{2} = it2ts{2};

iS2kpe{1} = uk(1:npe_a) ;
iS2kpe{2} = uk(end - npe_a +1 : end) ;

if s.verbose
    hS = figure('Name',['iS ',namestr]) ;
    plot(it2ts{1}, it2iS{1},'DisplayName','S1'),  hold on, grid
    plot(it2ts{2}, it2iS{2},'DisplayName','S2'), legend
    xlabel('it'), ylabel('iS'), drawnow
    
    hfull = figure('Name',['full ',namestr]) ;
    plot([1:npe_a], iS2ifull{1},'DisplayName','ifull 1'),  hold on, grid
    plot([1:npe_a], iS2ifull{2},'DisplayName','ifull 2')
    xlabel('iS'), ylabel('ifull')
    legend, drawnow
    
    hkpe = figure('Name',['kpe ',namestr]) ;
    plot([1:npe_a], iS2kpe{1},'DisplayName','kpe 1'),  hold on, grid
    plot([1:npe_a], iS2kpe{2},'DisplayName','kpe 2')
    xlabel('iS'), ylabel('kpe')
    legend, drawnow
    
    his2ts = figure('Name',['is2ts',namestr]) ;
    plot([1:npe_a], iS2ts{1},'DisplayName','ts 1'),  hold on, grid
    plot([1:npe_a], iS2ts{2},'DisplayName','ts 2')
    xlabel('iS'), ylabel('ts')
    legend, drawnow
    
    eshow(SMt)
end

% Below only for the non time-segmented method
PS_pe = 1 ; PS_fe =1; % pixel spacing

FOV_pe = PS_pe * npe_f ;
FOV_fe = PS_fe * nfe ;
fev  = ([1:nfe]-DCgp([1:nfe],2) ) ;
pefv = ([1:npe_f]-DCgp([1:npe_f],2)) ;

kfev = fev / FOV_fe ; % ksp
xpev = pefv * PS_pe ; % image coord
yfev = fev * PS_fe ;  % image coord


if mod(npe_u,2)~=1, warning(['Even npe_u']), end

% Scaling in CG:
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
% % mv_scale = sqrt(npe_f * nfe)   might be more correct, but it doesn't
% matter?


% Compute b (simulation of acquired k-space data)
B0curr = B0_act ;
b = afun(img(:), 'notransp') ;
disp(['Norm b (Ax=b) before noise: ',num2str(norm(b)) ])

% add noise in image domain (to get sensible scaling)
if s.addnoise
    % convert to image to add noise
    brs = reshape(b, [nfe, npe_a, ntraj, ncoil]) ;
    imbrs = k2isbs(brs) ;
    imbrs = addimnoise(imbrs, s.snr) ;
    b = i2ksbs(imbrs) ;
    clear imbrs  brs
    b = b(:) ;
end


disp(['Norm of b after noise added is: ', num2str(norm(b))])
disp(['Norm of img is: ',num2str(norm(img(:))) ])

    
B0_init = B0_act*s.B0_scale + s.B0_offset ; % Initial value in optimisation
disp(['Norm of B0_init: ',num2str(norm(B0_init(:)))])

B0curr = B0_init ;
img_check = afun(b(:), 'transp') ; % Conjugate Phase 
disp(['Norm of A^H b using B0_init is: ',num2str(norm(img_check)) ])

% FFT is unitary so CP gives a k2i type transform
% Note that when there are coil sensitivities, E is not unitary, i.e. E^H
% is not E^-1. In particular, there will be scaling differences 

figure(hf), subplot(sm, sn, 3)
imagesc(reshape(abs(img_check),[nfe npe_f])), colormap('gray')
colorbar, axis image
title('img check (CP)')

ftb = k2isbs(reshape(b,[nfe npe_a ntraj ncoil])) ;

subplot(sm, sn, 4)
srs = sqrt(sum((ftb.*conj(ftb)),4)) ;
imagesc(srs(:,:,1)), colorbar, axis image
title('SOS fft on b, traj 1')
if ntraj > 1
   subplot(sm, sn, 5), imagesc(srs(:,:,2)), title('SOS fft on b, traj 2')
   colorbar, axis image
end

switch s.x0
    case 'empty'
        x0=[] ;
case 'img'
        x0 = img ;
    otherwise
        error(['Unknown x0 option: ',s.x0])
end

if s.B0_joint && s.check_linear

    arg = -2i*pi*it2ts{1}(end)*(B0_act - B0_init) ; % use last time of traj1
    
    eshow(exp(arg) - (1 + arg), 'Name','Linearization error (itraj 1, final tseg)')
end


% Joint Estimation
xcurr = x0 ;

lsqr_str = {'converged to tol', 'max iterations', ...
    'pre cond was ill conditioned', 'stagnated','scalar too extreme'} ;

ixopt = 0 ;
ibopt = 0 ;

% Stores outputs from reconstructions for display as movie or montage using
% eshow
mvimg = zeros([nfe npe_f nouter+1]) ;
if ~isempty(xcurr), mvimg(:,:,1) = xcurr; end 

mvB0 =  zeros([nfe npe_f nouter+1]) ;
mvB0(:,:,1) = B0curr ;

for iouter = 1: nouter
    % first estimate image given B0
    disp(['Outer iteration number: ',num2str(iouter)])
    
    if s.x0_joint
        % Check forward and adjoint operations are consistent
        % < Du, v >=< u, DHv > 
        if iouter == 1
            v= rand(size(img(:))) ;
            Av = afun(v, 'notransp') ;
            u = rand(size(Av(:))) ;
            AHu = afun(u,'transp') ;
            if abs( dot(Av,u) - dot(v,AHu) ) > 0.001
                warning(['Dot products: ',num2str(dot(Av,u)), '  ', ...
                    num2str(dot(v,AHu))])
            end
        
            % check time segmentation vs DFT
            if s.x_check_tseg
                store_tseg = time_seg ;
                time_seg = true ;
                Avtst = afun(img(:), 'notransp') ;
                AHt   = afun(Avtst, 'transp') ;
                
                time_seg = false ;
                Avtsf =  afun(img(:), 'notransp') ;
                AHf    = afun(Avtst, 'transp') ;
                
                if norm(Avtst - Avtsf) > 0.01 % not exact!
                    warning(['fwd transforms disagree'])
                else
                    disp(['Passed check on fwd transform in afun'])
                end
                
                if norm(AHt - AHf) > 0.01
                    warning(['adjoint transforms disagree'])
                else
                    disp(['Passed check on adjoint transform in afun'])
                end
                
                time_seg = store_tseg;
            end
        end
        
        if ~isempty(xcurr)
            Ax = afun(xcurr(:),'notransp') ;
            rrnorm = (norm(b-Ax)/norm(b)) ;
            ixopt = ixopt+1 ; xopt(ixopt) = iouter-0.1; yopt(ixopt) = rrnorm ; 
            ytopt(ixopt) = norm(img(:)-xcurr(:))/norm(img(:)) ;
            disp(['Rel residual norm for x iter: ',num2str(rrnorm)])
        end
        
        % CG for image
        [x1, flag,relresx,iter,resvecx{iouter}] = lsqr(@afun, b, tol, ...
                                                maxit, @mfun,[],xcurr(:)) ;
        
        xcurr = reshape(x1, [nfe npe_f])  ;
        
        mvimg(:,:,iouter+1) = xcurr ;
        
        disp(['LSQR (im) finished after ',num2str(iter), ...
            ' iterations due to: ',lsqr_str{flag+1}, ...
            ', relres ',num2str(relresx), ...
            ', norm(x1): ',num2str(norm(x1))  ])
        
        % record rnorm after
        ixopt=ixopt+1 ; xopt(ixopt) = iouter+0.1; yopt(ixopt) = relresx ; 
        ytopt(ixopt) = norm(img(:)-xcurr(:))/norm(img(:)) ;
    end
    
    if s.B0_joint
        % See Matakos thesis, Chapter 4. Note B0 in my code if frequency and
        % so has to be multipleid by 2.pi to convert to omega.
        % In Matakos thesis, there is a "i" missing from the expression for
        % B between eqn 4.11 and 4.12
        
        % For Matakos linear scheme, input "k-space" is 
        % a difference    y' = y - A C f  + B B0est
       
        
        % Check residuals and transpose operations
        v = rand(size(B0curr(:))) ;
        Av = bfun(v, 'notransp') ;
        u = rand(size(Av(:))) ;
        AHu = bfun(u, 'transp') ;
        if abs( dot(Av,u) - dot(v,AHu) ) > 0.001 
            warning(['Dot products: ',num2str(dot(Av,u)), '  ', ...
                      num2str(dot(v,AHu))])
        end

        
        
        ibopt = ibopt+1; xbopt(ibopt)=iouter-0.1 ;
        ytbopt(ibopt) = norm(B0_act(:)-B0curr(:)) / norm(B0_act(:)) ;
        
        
        switch s.B0_opt
            case {'lsqr', 'LSQR'}
                % see below eqn 4.15
                bd = b - afun(xcurr(:), 'notransp') + bfun(B0curr(:), 'notransp') ;
        
                rrnorm = norm(bd- bfun(B0curr(:),'notransp')) / norm(bd) ;
                ybopt(ibopt)=rrnorm ;
                disp(['Rel resid norm for B0 iter: ',num2str(rrnorm) ])
                
                [x2, flag,relres,iter,resvec] = lsqr(@bfun, bd, tol, ...
                                             maxit, @mbfun,[],B0curr(:)) ;
                B0curr = reshape((x2), [nfe npe_f]) ;
                disp(['LSQR (B0): ',num2str(iter), ...
                      ' iterations due to: ',lsqr_str{flag+1}, ...
                      ', relres ',num2str(relres), ...
                      ', norm(x2): ',num2str(norm(x2)), '  norm(real(x2)): ',...
                       num2str(norm(real(x2))) ])
            case 'lsqnonlin'
                B0pre = B0curr ;
                options=optimoptions('lsqnonlin', 'Display','iter', 'UseParallel',true) ;
                [x2,resnorm] = lsqnonlin(@lsqnonlinfun, [0 1], [],[],options) ;
                relres = 0;
                B0curr = real(x2(2))*B0pre + real(x2(1)) ;
                disp(['x(2) ',num2str(x2(2)),' x(1) ',num2str(x2(1))])
            case 'fminsearch'
                B0pre = B0curr ;
               
                [x2,resnorm] = fminsearch(@fminsearchfun, [0 1]) ;
                
                B0curr = x2(2)*B0pre + x2(1) ;
                disp(['B0. x(2) ',num2str(x2(2)),' x(1) ',num2str(x2(1))])
            otherwise
                error(['Unknown s.B0_opt'])
        end
        
        mvB0(:,:,iouter+1) = B0curr ;
        
        if s.B0_project_real
            B0curr = real(B0curr) ;
        end
        
        
        ibopt = ibopt+1; xbopt(ibopt)=iouter+0.1; 
        ytbopt(ibopt) = norm(B0_act(:)-B0curr(:)) / norm(B0_act(:)) ;
        switch s.B0_opt
            case {'lsqr','LSQR'}
                ybopt(ibopt) = relres ;
        end
    end
end

eshow(cat(2,B0_act, B0_init,B0curr, B0curr-B0_act),'Name','B0  act init curr diff')
eshow(mvimg,'Name','img iterations')
eshow(mvB0, 'Name','B0 iterations')

figure(hf)

if s.x0_joint == true
    subplot(sm, sn, 6),
    for iouter = 1:nouter
        plot(resvecx{iouter}), hold on, title('x0 resvec')
    end  
    xlabel('Iter'), ylabel('residual')
    
    im_out = reshape(x1,[nfe npe_f]) ;
    rnorm = norm(im_out-img)/norm(img) ;
    disp(['Rel norm of final difference: ',num2str(rnorm)])
    figure(hf), subplot(sm, sn, 9), imagesc(abs(im_out)), title('im_out')
    colorbar , axis image
    eshow(cat(2,img, im_out, (im_out - img)),'Name',[s.name,' img, im_out, diff'])
else
    rnorm = 0 ;
end


figure(hf), subplot(sm, sn, 8), imagesc(abs(img)), title('img'), axis('image')
colorbar
if nouter > 1
    figure('Name',['rnorms ', namestr]), plot(xopt, yopt, 'DisplayName','Im')
    grid, hold on
    plot(xopt, ytopt,'LineWidth',2,'DisplayName','true Im')
    if s.B0_joint
        switch s.B0_opt
            case {'LSQR','lsqr'}
                plot(xbopt, ybopt,'DisplayName','B0')
        end
        plot(xbopt, ytbopt,'LineWidth',2, 'DisplayName','true B0')
    end
    xlabel('Outer iter'), ylabel('Residual Norm'), legend
end

% afun 
    function y = afun(x, transp_flag)
        if strcmp(transp_flag,'notransp') % FWD model one image to k-spaces
            % S = Asamp F C Im
            % K from I, integral over I
            S = zeros([nfe, npe_a, ntraj, ncoil]) ;
            imgx = reshape(x,[nfe npe_f]) ;
            if time_seg == true
                for itraj  = 1:ntraj
                    for icoil = 1:ncoil
                        for itseg = 1:ntseg
                            indS = it2iS{itraj}(itseg) ;
                            ts = iS2ts{itraj}(indS) ;
                            
                            St = kfi(csens(:,:,icoil) .* exp(-2i*pi*B0curr * ts) .* imgx ) ;
                            % here, St is 2D k-space from fully sampled image
                            % For SENSE, now reduce size.
                            % We are interpolating in PE direction (here
                            % along rows). Interpolate, the take out those
                            % indexed by pes
                            
                            % St is [nfe npe_f]
                            % St.' is [npe_f  nfe]
                            % The sinc interpolation generates Sus [nfe  npe_u]
                            
                            Sus = transpose(SMt * transpose(St)) ;
                            
                            S(:,indS,itraj, icoil) = Sus(:,iS2ifull{itraj}(indS)) ;
                        end % itseg
                    end % icoil
                end % itraj
            else
                % afun, no time segmentation used in recon
                for icoil = 1:ncoil
                    for itraj = 1:ntraj
                        for ia = 1:npe_a  % indexing time 
                            for ikfe = 1:nfe
                                for iipe = 1:npe_f
                                    for iife = 1:nfe
                                        indS = it2iS{itraj}(ia) ;
                                        kpeS = iS2kpe{itraj}(indS) / FOV_pe ;
                                        S(ikfe,indS,itraj,icoil) = ...
                                            S(ikfe,indS, itraj, icoil) + ...
                                            csens(iife, iipe, icoil) * ...
                                            imgx(iife, iipe) * ...
                                            exp(-2i*pi*(kpeS*xpev(iipe) + ...
                                                kfev(ikfe)*yfev(iife) + ...
                                                B0curr(iife, iipe)*it2ts{itraj}(ia))) ;  
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
            S = S / mv_scale ;
            y = S(:) ;
            
        elseif strcmp(transp_flag,'transp') % afun, A^H y K-spaces to image IFK
            
            W = zeros([nfe npe_f]) ;
            Sx = reshape(x,[nfe npe_a ntraj ncoil]) ; 
            
            if time_seg == true
                for itraj  = 1:ntraj
                    for icoil = 1:ncoil
                        for itseg = 1:ntseg
                            indS = it2iS{itraj}(itseg) ;
                            ts = iS2ts{itraj}(indS) ;
                            
                            % Need to get bits out of Sx and assemble.
                            
                            Stemp = zeros([nfe npe_a]) ;
                            Stempu = zeros([nfe npe_u]) ;
                            Stemp(:,indS) = Sx(:,indS,itraj,icoil) ;
                            
                            % first and last transposes are for shape of
                            % data. Middle is for adjoint operation
                            % Stemp is [nfe npe_a]
                            % Put into [nfe npe_u] prior to interpolation
                            
                            Stempu(:,iS2ifull{itraj}(:)) = Stemp ;
                            % Need to get Stempf at [nfe npe_f]
                            Stempf = transpose( transpose(SMt) * transpose(Stempu)) ;
                            
                            W = W + exp(2i*pi*B0curr * ts) .* ...
                                    conj(csens(:,:,icoil)) .* ...
                                    ifk( Stempf ) ;
                        end 
                    end
                end
            else
                % No time_seg in reconstruction (afun)
                for itraj = 1:ntraj
                    for ia = 1:npe_a
                        indS = it2iS{itraj}(ia) ;
                        kpeS = it2ka(ia,itraj) / FOV_pe ;
                        for ikfe = 1:nfe
                            for iipe = 1:npe_f
                                for iife = 1:nfe
                                    for icoil = 1:ncoil
                                        % Integral over k
                                        W(iife,iipe) = W(iife,iipe) + ...
                                            conj(csens(iife, iipe,icoil)) * ...
                                            Sx(ikfe, indS, itraj,icoil) * ...
                                            exp(2i*pi*(kpeS*xpev(iipe) + ...
                                            kfev(ikfe)*yfev(iife) + ...
                                            B0curr(iife, iipe)*it2ts{itraj}(ia) )) ; 
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
            y = W(:) / mv_scale ;
        end
    end

% LSQNONLIN approach
    function F = lsqnonlinfun(B0scale)
        B0curr = real(B0scale(2))*B0pre  + real(B0scale(1)) ;
        
        F = b - afun(xcurr, 'notransp') ;
    end

% FMINSEARCH approach
    function F = fminsearchfun(B0scale)
        B0curr = B0scale(2)*B0pre  + B0scale(1) ;
        
        F = b - afun(xcurr(:), 'notransp') ;
        
        F = sum(F .* conj(F)) ;
    end


% pre-condition - certainly helps in simulation
    function y= mfun(x, transp_flag)
        W = reshape(x,[nfe npe_f]) ;
        
        if strcmp(transp_flag,'notransp') % 'notransp') returns M\x
            % If FWD model A is m x n, then M should be n x n. So, n must
            % be image size and input x must be image
            cw = pcsos .* W ;
        elseif strcmp(transp_flag,'transp') % returns M'\x.
            cw = conj(pcsos) .* W ; % transpose is of full diag matrix. pcsos is square.
        end
        
        y = cw(:) ;
    end

    function y = mbfun(x, transp_flag) % preconditioning for B0 estimation
        W = reshape(x,[nfe npe_f]) ;
       % cx = csos .* xcurr ;
        cx = csos ;
        acxmax = max(abs(cx(:))) ;
        lowcx = abs(cx) < acxmax * pc_thresh ;
        cx(lowcx) = acxmax * pc_thresh ;
        pc = 1./(cx) ;
        
        if strcmp(transp_flag,'notransp') % 'notransp') returns M\x
            % If FWD model A is m x n, then M should be n x n. So, n must
            % be image size and input x must be image
      
            cw = pc .* W ;
        elseif strcmp(transp_flag,'transp') % returns M'\x.
            cw = conj(pc) .* W ; % transpose is of full diag matrix. pcsos is square.
        end
        
        y = cw(:) ;
    end


     function y = bfun(x, transp_flag) % For B0 estimate in joint recon
        if strcmp(transp_flag,'notransp') % FWD model one image to k-spaces
            % y' = B B0new
            %    y' = y - A C f  + B B0curr
            % where f is image vector, F is diag(f), A is usual system matrix 
            % using B0curr and 
            %  B is  - diag(t) A diag(c) diag(f)
            %
            % See Matakos thesis near equation 4.11
            S = zeros([nfe, npe_a, ntraj, ncoil]) ;
            
            B0x = reshape(x,[nfe npe_f]) ;
            if time_seg == true
                for itraj = 1:ntraj
                    for icoil = 1:ncoil
                        for itseg =1:ntseg
                            indS = it2iS{itraj}(itseg) ;
                            ts = iS2ts{itraj}(indS) ;
                            
                            St = -2i*pi *ts * ...
                                kfi(exp(-2i*pi*B0curr * ts) .* ...
                                    csens(:,:,icoil) .* xcurr .* B0x ) ;
                            Sus = transpose( SMt * transpose(St)) ;
                            S(:,indS,itraj, icoil) = Sus(:,iS2ifull{itraj}(indS) ) ;
                        end % itseg
                    end % icoil
                end % itraj
            else
                for icoil = 1:ncoil
                    for itraj = 1:ntraj
                        for ia = 1:npe_a
                            for ikfe = 1:nfe
                                for iipe = 1:npe_f
                                    for iife = 1:nfe
                                        indS = it2iS{itraj}(ia) ;
                                        kpeS = iS2kpe{itraj}(indS) / FOV_pe ;
                                        ts_this = it2ts{itraj}(ia) ;
                                        S(ikfe,indS,itraj,icoil) = ...
                                            S(ikfe,indS, itraj, icoil) + ...
                                            xcurr(iife,iipe) * ...
                                            csens(iife, iipe, icoil) * ...
                                            exp(-2i*pi*(kpeS*xpev(iipe) + ...
                                                kfev(ikfe)*yfev(iife) + ...
                                                B0curr(iife, iipe)*ts_this)) * ...
                                            -2i*pi*ts_this * B0x(iife,iipe) ;
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
            S = S / mv_scale ;
            y = S(:) ;
            
        elseif strcmp(transp_flag,'transp') % A^H y     K-space diff to image (B0)
            % %  B is  - diag(t) A diag(c) diag(f)
            % Transpose operation is then (with diag replaced by capitals)
            %  - F^H C^H A^H diag(t)
            %  ^^^^^^^^^ image      ^^^ k-space
            
            W = zeros([nfe npe_f]) ; % this will be the new B0map
            Sx = reshape(x,[nfe npe_a ntraj ncoil]) ; 
            
            if time_seg == true
                for itraj  = 1:ntraj
                    for icoil = 1:ncoil
                        for itseg = 1:ntseg
                            indS = it2iS{itraj}(itseg) ;
                            ts = iS2ts{itraj}(indS) ;
                            
                            % Need to get bits out of Sx and assemble.
                            Stemp = zeros([nfe npe_a]) ;
                            Stempu = zeros([nfe npe_u]) ;
                            Stemp(:,indS) = Sx(:,indS,itraj,icoil) ;
                            Stempu(:,iS2ifull{itraj}(:)) = Stemp ;
                            Stempf = transpose( transpose(SMt) * transpose(Stempu) ) ;
                            W = W + conj(xcurr) .* ...
                                    conj(csens(:,:,icoil)) .* ...
                                    exp(2i*pi*B0curr * ts) .* ifk( +2i * pi * ts * Stempf ) ;
                        end
                    end
                end
            else
                % No time_seg in reconstruction
                for itraj = 1:ntraj
                    for ia = 1:npe_a
                        indS = it2iS{itraj}(ia) ;
                        kpeS = it2ka(ia,itraj) / FOV_pe ;
                        for ikfe = 1:nfe
                            for iipe = 1:npe_f
                                for iife = 1:nfe
                                    for icoil = 1:ncoil
                                        % Integral over k
                                        
                                        W(iife,iipe) = W(iife,iipe) + ...
                                            2i*pi*it2ts{itraj}(ia) * ...
                                            exp(2i*pi*(kpeS*xpev(iipe) + ...
                                            kfev(ikfe)*yfev(iife) + ...
                                            B0curr(iife, iipe)*it2ts{itraj}(ia) )) * ...
                                            conj(csens(iife, iipe,icoil)) * ...
                                            conj(xcurr(iife,iipe)) * ...
                                            Sx(ikfe, indS, itraj,icoil)  ;
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
            y = W(:) / mv_scale ;
        end
    end
end

function img = ifk(ksp)
% IFK image from k-space DFT without scaling.
% Note for img2 = ifk(kfi(img)), img2 will have different scale to img
%
% MATLAB's  fft has -ve exp and no scaling
%          ifft has +ve exp and is scaled by 1/n
%
% My old i2k uses the MATLAB fft (thought to be incorrect, but
% Matakos thesis uses -ve exp for kfi (i2k) )
%
%  K <- I  (kfi) FWD model  -ve exp,  MATLAB FFT
%  I <- K  (ifk) ^H    
% 

[ny, nx, nz] = size(ksp) ;
if nz ~= 1
    error(['Expecting 2D input to ifk'])
end

img = nx * ny * fftshift( ifftn( ifftshift( ksp ) ) ) ;

end

function ksp = kfi(img)
% KFI i2k with scaling removed
% See IFK

ksp = fftshift( fftn( ifftshift( img) ) ) ;

end

        
