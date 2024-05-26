function [yout, opt] = sysmatv(xin, transp_flag, opt)
% SYSMATV Product of system matrix and vector, or adjoint (Hermitian transpose)
% Intended for use with LSQR and also as a discrete Fourier Transform
% Implements b = Ax or m = A^H v
%
%  yout = sysmatv(xin, transp_flag, opt)
%  [yout, opt] = ...
%
%  To set opt using defaults, run with fwd model:
%     [yout, opt] = sysmatv(xin, 'notransp', optinit)   or
%                 = sysmatv(xin, 'notransp') 
%
%  For LSQR, xin and yout must be column vectors
% 
% transp_flag has value 'notransp' or 'fwd' (foward model),  'transp' or 'adj' 
%             for adjoint, or, 'parse' to just set opt
%             'selectyout' to chop yout to just data for Rs2use
%
% opt.verbose {true} false
% opt.ntraj - number of trajectories, defaults to 1.
% opt.ncoil - OUTPUT ONLY number of coils, defaults to 1.
% opt.szxf - matrix size of xin in forward model. Defaults to size(xin).
% opt.csens - [szxf ncoil] coil sensitivities. Defaults to ones(szxf)
% opt.sens_lower_thresh  def 0.01 threshold of max sos sensitivity to use in
%                        preconditioning to prevent divide by zero.
% opt.pcsos  - SOS for preconditioning
% opt.Rreq - Desired speed up factor. Can be a vector for testing idea 
%            of using different values. Default 1.
% opt.Ract - OUTPUT ONLY Actual speed up applied
%
% opt.tstep  time between lines corresponding to successive indices
%            into k-space coordinates (defaults to 1ms) size nR. Use
%            negative value to indicate reverse PE direction. tstep should
%            correspond to acquisition and not be adjusted for skipped
%            lines when parallel imaging
% opt.szyf - matrix size for yout in forward model. Defaults to 
%            [szxf(1)/R ntraj ncoil], will be vectorised if opt.vec true
%            If not supplied, Rreq will be applied. Size does not change
%            with partial Fourier - applied using mask.
%            If R is a vector, first dimension will szxf(1)/R(1) +
%            szxf(1)/R(2) + ...
% opt.nkperR, opt.ckstart, opt.ckend OUTPUT only, set when no opt.szyf
% opt.partialFReq default 1. value between 0.5 and 1
% opt.pFskip      +1 or -1. Default -1. Which part of k-space to mask for partial
%                 Fourier size nR.
% opt.partialFAct Output. Actual partial Fourier factor.
%                 Partial Fourier is applied by zeroing k-space after fwd
%                 transform, and on input to adj. First indices are zerod.
% opt.mp   - [szxf nR]  motion induced phase in image domain, per R.
% opt.Rs2use [nR 1] defaults to all Rs
% opt.FOV  - FOV (mm). Defaults to size of image (opt.szxf).
% opt.cr1  - coordinates of image 1st dimension. Defaults to -FOV/2 to 
%            FOV/2 with appropriate care for even/odd arrays.
% opt.ck1  - coordinates of output k-space in fwd model. Defaults to 
%            [-1/(2.FOV) ... 1/(2.FOV)]*R with appropriate edge and odd/even.
%            If not supplied, Rreq will be taken into accout when
%            calculating ck1 (R will have value Ract).
%            If Rreq is vector, coordinates are concatenated
%            Coordinates increase in the array for every R, to indicate reverse travel
%            through k-space, use a negative time step for that R.
% opt.vec  - {false} Should be set true for LSQR
% 
% opt.fmodsign {-1}, +1  sign in exponential for forward model
%    Note MATLAB FFT uses -ve sign and scale of 1, iFFT uses +ve sign and 1/N scale
% opt.scale {'balanced'}, 'i2k', 'step'
%   for conjugate gradient use 'balanced' 1/sqrt(N).
%   'i2k' matches the i2k function (uses MATLAB fft)
%   'step' uses the integration step size - a truer value
%
% opt.b0fun function that returns B0 in Hz given position as input. Default
% zero
% opt.Tfun  function that returns time in s given traj and ksp line as
% input. Default 0
%
% opt.precond  {false} use preconsiditioning based on coil sos
% opt.maxit    stores a maxit for lsqr
%
% Implements Ax=b or A^H v = m
%
% Internal function variables and sizes:
%                     xin                      yout
% FWD      Ax = b     x [ncr1 1]              b [nck1 ntraj ncoil]
% ADJ   A^H v = m     v [nck1 ntraj ncoil]    w [ncr1 1]
%
% Dimensions for b and v are "k-space" and for x and w are "image".
%

if nargin < 3
    opt = [] ;
end

[opt, fscale, ascale] = parse_opt(xin, opt) ;    

nck1 = length(opt.ck1) ;
ncr1 = length(opt.cr1) ;

ntraj = opt.ntraj ;
ncoil = opt.ncoil ;

fsign = opt.fmodsign ;
B0_Hz = opt.b0fun ;
T = opt.Tfun ;
csens = opt.csens ;
Rs2use = opt.Rs2use ;
cksthis = [] ;
for iR2u = 1:length(Rs2use)
    cksthis = cat(1,cksthis, [opt.ckstart(Rs2use(iR2u)):opt.ckend(Rs2use(iR2u))]') ;
end
szB = opt.szyf ;
szB(1) = length(cksthis) ;

switch transp_flag
    case {'notransp','fwd'}
        %   FWD model i.e. image to k-space
        % Matrix representation B = F C B0 P X
        % When more than 1D, string X into a vector
        % Successive measures are strung out in the first dimenson of B.
        % When called for just one measure (or R), need to extract the
        % relevant portion.
        
        B = zeros(szB) ; % internal 'kspace'
        img = xin(:) ;
        
        for icr1 = 1:ncr1
            B0_this = B0_Hz(icr1) ;
            for icksthis = 1: length(cksthis)
                ick1 = cksthis(icksthis) ; % index into full set of coordinates
                for icoil = 1:ncoil
                    for itraj= 1: ntraj
                        B(icksthis, itraj, icoil) = B(icksthis, itraj, icoil) + ...
                            csens(icr1,icoil) * opt.mp(icr1, opt.ick2R(ick1)) * img(icr1) * ...
                            exp(fsign*2i*pi*(opt.ck1(ick1)*opt.cr1(icr1) + (B0_this * T(itraj, ick1))) ) ;
                    end
                end
            end
        end
        B = B * fscale ;
        
        B = B .* opt.kmask(cksthis) ;  % uses implicit expansion of mask
        
        if opt.vec
            yout = B(:) ;
        else
            yout = B ;
        end
        
    case {'transp','adjoint'}  % transp i.e. complex transpose (adjoint) operation
        %  m = A^H v
        
        % detect if full k-space passed in, or just that needed for the
        % measures under consideration.
        nd1 = numel(xin) / ncoil / ntraj ;
        if nd1 > length(cksthis)
            V = reshape(xin, [nd1 ntraj ncoil]) ;
            V = V(cksthis,:,:) ;
        else
            V = reshape(xin, [length(cksthis) ntraj ncoil]) ; % input  "kspace"
        end
        
        M = zeros([ncr1 1]) ;                  % output "image"
        
        V = V .* opt.kmask(cksthis) ;
        
        for icr1 = 1: ncr1
            B0_this = B0_Hz(icr1) ;
            for itraj = 1: ntraj
                for icoil = 1: ncoil
                    for icksthis = 1: length(cksthis)
                        ick1 = cksthis(icksthis) ;
                        M(icr1) = M(icr1) + ...
                            V(icksthis, itraj, icoil) * ...
                            conj(csens(icr1,icoil)) * ...
                            conj( opt.mp(icr1, opt.ick2R(ick1)) ) * ...
                            exp(-fsign*2i*pi*(opt.ck1(ick1)*opt.cr1(icr1) + ...
                            (B0_this * T(itraj, ick1))) ) ;
                    end
                end
            end
        end
        
        yout = M * ascale ;
        if opt.vec
            yout = yout(:) ;
        end
    case 'parse'
        % do nothing
        yout = [] ;
    case 'selectyout'
        nd1 = numel(xin) / ncoil / ntraj ;
        if nd1 > length(cksthis)
            V = reshape(xin, [nd1 ntraj ncoil]) ;
            V = V(cksthis,:,:) ;
        else
            V = reshape(xin, [length(cksthis) ntraj ncoil]) ; % input  "kspace"
        end
        yout = V ;
        if opt.vec
            yout = yout(:) ;
        end
        
    otherwise
        error('Unknown transp_flag')
end

end % end sysmatv

% PARSE_OPT
function [opt, fscale, ascale] = parse_opt(xin, opt)
if size(xin,2) > 1
    error('Only 1D at present')
end

allowedfields = {'vec', 'ntraj', 'szxf', 'csens', 'Rreq', 'pFskip', 'Ract', 'tstep', ...
    'partialFReq', 'FOV', 'cr1', 'ck1', 'fmodsign', 'scale', 'b0fun', ...
    'mp', 'ick2R', 'Rs2use', 'sens_lower_thresh', 'pcsos', ...
    'Tfun', 'verbose' , 'precond', 'maxit', ...
    'ncoil', 'nkperR', 'ckend', 'ckstart', 'szyf', 'kmask','partialFAct', 'ckind','ckt'} ;
 
fopt = fields(opt) ;
tf = contains(fopt , allowedfields) ;
if ~all(tf)
    warning(['Unknown field: ', fopt{tf==false}])
end

if ~isfield(opt, 'vec')
    opt.vec = false ;
end

if ~isfield(opt,'ntraj')
    opt.ntraj = 1 ;
end


if ~isfield(opt, 'szxf')
    opt.szxf = size(xin) ;
end

if ~isfield(opt, 'csens')
    opt.csens = ones(opt.szxf) ;
end

opt.ncoil = size(opt.csens,2) ;

if ~isfield(opt, 'sens_lower_thresh')
    opt.sens_lower_thresh = 0.01 ;
end

if ~isfield(opt, 'pcsos') 
    csos = sqrt(sum(opt.csens.*conj(opt.csens), 2)) ; 
    csos_max = max(csos(:)) ;
    low = csos < csos_max*opt.sens_lower_thresh ;
    csos(low) =  csos_max*opt.sens_lower_thresh ;

    opt.pcsos = 1./csos ;   
end

if ~isfield(opt, 'Rreq')
    opt.Rreq = 1 ;
end
nR = length(opt.Rreq) ;

if ~isfield(opt, 'Rs2use')
    opt.Rs2use = [1:nR] ; % use all by default
end
Lia = ismember(opt.Rs2use, [1:nR]) ;
if ~all(Lia)
    error('Rs2use must be in Rs')
end

if ~isfield(opt, 'tstep')
    opt.tstep = ones(size(opt.Rreq)) * 1e-3 ;
end

if length(opt.tstep) == 1 && nR>1
    opt.tstep = repmat(opt.tstep, [nr 1]) ;
end


if ~isfield(opt, 'szyf')
    nx1 = opt.szxf(1) ;
    nxtot = 0 ;
    opt.nkperR = [];
    for iR = 1 : nR
        nx1R = round(nx1/opt.Rreq(iR)) ;
        nxtot = nxtot + nx1R ;
        opt.nkperR(iR) = nx1R ;
        opt.ckend(iR) = sum(opt.nkperR) ;
        opt.ckstart(iR) = opt.ckend(iR) - opt.nkperR(iR) + 1 ;
        opt.Ract(iR) = nx1/nx1R ;
        opt.ick2R(opt.ckstart(iR):opt.ckend(iR)) = iR ;
    end
    opt.szyf = [nxtot opt.ntraj opt.ncoil] ;
end

if ~isfield(opt, 'partialFReq')
    opt.partialFReq = 1;
end

if ~isfield(opt, 'pFskip')
    opt.pFskip = -1 ;
end

if length(opt.pFskip)>1 
    if length(opt.pFskip) ~= nR
        error(['pFskip should be scalar or nR elements'])
    end
else
    opt.pFskip = repmat( opt.pFskip, [nR 1] ) ;
end

opt.kmask = ones([opt.szyf(1) 1]) ;
for iR = 1:nR
    n2z = round( (1 - opt.partialFReq) * opt.nkperR(iR) ) ;
    opt.partialFAct(iR) = 1 - n2z/opt.nkperR(iR) ;
    
    if opt.pFskip(iR) == -1
        opt.kmask(opt.ckstart(iR):opt.ckstart(iR)+n2z-1) = 0 ;
    elseif opt.pFskip(iR) == +1
        opt.kmask(opt.ckend(iR)-n2z+1:opt.ckend(iR)) = 0 ;
    else
        error(['opt.pFskip should have value +1 or -1'])
    end
    
end

if ~isfield(opt, 'mp')
    opt.mp = ones(opt.szxf(1), nR) ;
else
    if size(opt.mp,2) ~= nR || size(opt.mp,1) ~= opt.szxf(1)
        error('opt.mp size problems')
    end
end

if ~isfield(opt,'FOV')
    opt.FOV = opt.szxf ;
end

if ~isfield(opt,'cr1')
    FOV1 = opt.FOV(1) ;
    N1 = opt.szxf(1) ;
    opt.cr1 = ( [1:N1] - sz2DC(N1) ) * FOV1/N1 ;
end

if ~isfield(opt,'ck1')
    FOV1 = opt.FOV(1) ;
    for iR = 1: nR
        N1k = opt.nkperR(iR) ;
        ckthisR = ( [1:N1k] - sz2DC(N1k)) * opt.Ract(iR) / FOV1 ;
        
        opt.ck1(opt.ckstart(iR):opt.ckend(iR)) = ckthisR ;
        
        ckind = [1:N1k] - sz2DC(N1k) ; 
        opt.ckind(opt.ckstart(iR) : opt.ckend(iR)) = ckind ;
        opt.ckt(opt.ckstart(iR) : opt.ckend(iR)) = ckind * opt.tstep(iR) ;
    end
    
    if nR == 1 
        if N1k ~= opt.szyf(1) ; error('something wrong'); end
    end
    
end

if ~isfield(opt, 'fmodsign')
    opt.fmodsign = -1 ;
else
    if abs(opt.fmodsign)~=1
        error('Sign must be +1 or -1')
    end
end

if ~isfield(opt, 'scale')
    opt.scale = 'balanced' ;
end

switch opt.scale
    case 'balanced'
        fscale = 1/sqrt(opt.ntraj * opt.ncoil * opt.szxf(1)) ;
        ascale = fscale ;
    case 'i2k'
        fscale = 1 ;
        ascale = 1/(opt.ntraj * opt.ncoil * opt.szxf(1)) ;
        if opt.fmodsign ~= -1
            warning('For i2k emulation, expecting sign to be -1')
        end
        if nR > 1
            warning('i2k scale not checked for nR > 1')
        end
    case 'step'
        fscale = opt.cr1(2) - opt.cr1(1) ;
        ascale = opt.ck1(2) - opt.ck1(1) ;
        if nR > 1
            warning('step scale not checked for nR > 1')
        end
    otherwise
        error('unknown scale')
end

if ~isfield(opt, 'b0fun')
    opt.b0fun = @(x) 0 ;
end

if ~isfield(opt, 'Tfun')
    opt.Tfun = @(x,y) 0 ;
end

if ~isfield(opt, 'precond')
    opt.precond = false ;
end

if ~isfield(opt, 'maxit')
    opt.maxit = [] ;
end

if ~isfield(opt, 'verbose')
    opt.verbose = true ;
end

if opt.verbose
    tfun = opt.Tfun ;
    
   disp(' ')
   pix = opt.cr1(2)-opt.cr1(1) ;
   disp(['FOV ',num2str(opt.FOV(1)),'mm. voxel: ',num2str(pix)])
   disp(['ncoil ',num2str(opt.ncoil),'. ntraj ',num2str(opt.ntraj)])
   disp(['Rreq ',num2str(opt.Rreq)])
   disp(['Partial Fourier Req ',num2str(opt.partialFReq)])
   for iR = 1:nR
       disp(['  Ract ',num2str(opt.Ract(iR))])
       disp(['  partial Fourier act ',num2str(opt.partialFAct(iR))])
       disp(['  k-space coords  ck1(',num2str(opt.ckstart(iR)),' ... ', ...
           num2str(opt.ckend(iR)),') = ',num2str(opt.ck1(opt.ckstart(iR))),' ... ',...
           num2str(opt.ck1(opt.ckend(iR))) ])
       tstep = tfun(1,opt.ckstart(iR)+1)-tfun(1,opt.ckstart(iR)) ;
       disp(['  Time step ', num2str(tstep),' s'])
       
       % In a SENSE recon where you FT the undersampled k-space and unfold later,
       % the FOV and N scale with R, leaving the voxel size unchanged. (FOV is
       % reduced giving the wrapping, N is reduced saving time)
       % When written as a fwd model, a full size FOV and N are maintained. Need
       % to adjust tstep here (?) to account for time it would take to jump a
       % k-space step of 1 unit i.e. 1/(full fov)
       b0off = 1/(tstep/opt.Ract(iR))/length(xin) ;
       disp(['B0 offset corresponding to 1 pixel (', ...
           num2str(pix),'mm) : ', ...
           num2str(b0off),' Hz. ( ',num2str(b0off/pix),' Hz/mm).'])
   end
   
   disp(['scale FWD ',num2str(fscale),'  ADJ ',num2str(ascale)]) 
end

end
    