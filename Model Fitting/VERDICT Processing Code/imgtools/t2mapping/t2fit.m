function dfit = t2fit(data, techo, varargin)
% T2FIT Fit T2 and luminal water fraction to data
%   Aim is to provide solid reference code 
%
% dfit = t2fit(data, techo, Name, Value, ...)
% dfit = t2fit(data, techo, props)
%
% data multi-echo data, last dimension must be necho
%      data converted to modulus double on input.
%
% techo  Echo Time. Either a scalar echo time difference, or, the 
%        each of the necho values (s)
%
% David Atkinson,  University College London
% D.Atkinson@ucl.ac.uk
%
% Code follows work of William Devine 
%
% See also 

% Default fit methods to use
fmeth = struct('mono', false, 'bi', true, 'lwi', false) ;
lwi_startest = 'nonlwi' ;

data = double(abs(data)) ; % convert to modulus double in case complex or single
szdata = size(data) ;
szmap = szdata(1:end-1) ; % Maps are same size as image component of data.

necho = szdata(end) ;        % number of echoes
np = prod(szdata(1:end-1)) ; % number of pixels in data
rdata = reshape(data, [np necho]) ; % reshape to make looping and parfor easier

% Default mask
mask = ones([np 1]) ;

loc_mask = find(mask) ;
nloc = length(loc_mask) ;

mdata = rdata(loc_mask,:) ;  % just the pixels in the mask (helps parfor)

if length(techo) > 1
    TEs = techo ;
else
    TEs = [1:necho].*techo ;
end

if TEs(1) > 1
    warning(['Expecting TEs in s, dividing by 1000'])
    TEs = TEs ./ 1000 ;
end

if fmeth.lwi
    % For lwi, probability distribution, number of points and maximum
    nprt2s = 1000 ;
    prt2s_max = 2000e-3 ; % now in s to match Devine
    
    prt2s = linspace(0,prt2s_max,nprt2s) ; % could also try logspace
    
    % Probability distribution - two gaussians characterised by 's'
    % s(1) overall amplitude
    % s(2) proportion of gaussian A
    % s(3) and s(5) mean and std of gaussian A
    % s(4) and s(6) mean and std of gaussian B
    pr = @(s) s(1).*((s(2).*normpdf(prt2s',s(3),s(5)))+((1-s(2)).*normpdf(prt2s',s(4),s(6)))) ; 
    A = exp(-TEs' * (1./prt2s) ); % implicit expansion to size [necho  nprt2s]

    std_max =  10e-3 ; % upper bound for Gaussian std. Will Devine had 10ms!
    t2bound = 200e-3 ; % T2 boundary (s) between short and long T2s.
    
    lb = [0,   0, 0,       t2bound,    0,        0 ] ;
    ub = [inf, 1, t2bound, max(prt2s), std_max, std_max] ;
    
end


std_init = 5e-3 ; % used in Devine and fixed. Do not change for Devine

% if Devine starting estimate, need to loop through slices, find median and
% fit to bi exponential
switch lwi_startest
    case 'Devine'
        % Does a bi-exponential fit on the median for each slice, then
        % a lsqnonlin and uses these as starting estimates.
        
        if length(szdata) == 3
            nz = 1 ;
        else
            nz = szdata(3) ;
        end
        
        bifopt = fitoptions('exp2', 'Lower', [0, -inf, 0, -inf], ...
                                  'Upper', [inf, -1/prt2s_max , inf,-1/prt2s_max], ...
                                  'Display', 'off') ;
                              
        dX0 = zeros([szdata(1) szdata(2) nz 6]) ;
        sig_med = zeros([nz, necho]) ;
        
        for iz = 1: nz
            if length(szdata) == 3
                sld = data ;
            else
                sld = squeeze(data(:,:,iz,:)) ;
            end
            for iecho = 1:necho
                esig = sld(:,:,iecho) ;
                sig_med(iz,iecho) = nanmedian(esig(:)) ;
            end
            
            bimedfit = fit(TEs(:), sig_med(iz,:)', 'exp2', bifopt) ;
            
            slx0(1) = bimedfit.a + bimedfit.c ./ 100 ; 
            slx0(2) = bimedfit.a ./ ( bimedfit.a + bimedfit.c ) ;
            slx0(3) = min(-1/bimedfit.b, t2bound) ;
            slx0(4) = max(-1/bimedfit.d, t2bound) ;
            slx0(5) = std_init ;
            slx0(6) = std_init ;
            
            % fix slx0(1) 
            slx0(1) = 1 ;
            sig = A*pr(slx0) ;
            slx0(1) = sig_med(iz,1) / sig(1) ;
            
            % Now do a pre-LWI fit (!)
            funsl = @(s) (A*pr(s))-squeeze(sig_med(iz,:)') ;
            sllsq = lsqnonlin(funsl, slx0, lb, ub) ;
            
            disp(['bi-fit: a ',num2str(bimedfit.a),' T2A ',num2str(-1000/bimedfit.b), ...
                ', c ',num2str(bimedfit.c),' T2B ',num2str(-1000/bimedfit.d),' ms'])
            
            disp(['after converting for lsqnonlin:'])
            disp_s(slx0)
            disp(['after lsqnonlin:'])
            disp_s(sllsq)
            
            dX0(:,:,iz,:) = repmat(reshape(sllsq,[1 1 1 6]),[szdata(1) szdata(2) 1 1]) ;
        end
        
        dX0 = reshape(dX0, [np 6]) ;
        mX0 = dX0(loc_mask,:) ;
            
        options = optimoptions('lsqnonlin','Display','none' );
        
    case 'fixed'
        % Identical, fixed starting estimates for Gaussians 's' 
        % X0(1) = median(mdata(:,1)) ;   % amplitude
        X0(1) = 1 ;  % sort below
        X0(2) = 0.95 ;                  % 1-lwf
        X0(3) = 30e-3 ; X0(5) = std_init ;     % mean and std
        X0(4) = 400e-3 ; X0(6) = std_init ;    % mean and std
        
        sig = A*pr(X0) ;
        X0(1) = median(mdata(:,1)) / sig(1) ;
        
        mX0 = repmat(X0(:)', [nloc 1]) ; % [nloc 6]
        
        options = optimoptions('lsqnonlin','Display','none','TypicalX',X0 );
    case 'nonlwi'
    otherwise
        error(['Unknown starting_estimate value'])
end




% Check TEs are equally spaced (allow for small variations from DICOM.)
echo_thresh = 0.1e-3; % threshold for echo time separation difference (0.1 ms)
diffTEs = diff(TEs) ;
loc_uneven = find(abs(diffTEs - diffTEs(1)) > echo_thresh ) ;
if ~isempty(loc_uneven)
    warning(['Echo time separations differ by more than ',num2str(echo_thresh)])
end

% Preparation for parfor
% Note needed to place only masked pixels in mdata to help parallel aspect
do_mono = false ; do_bi = false ; do_lwi = false;

if fmeth.mono,  do_mono = true; fmono = cell(nloc, 1);  end 
if fmeth.bi,    do_bi = true;   fbi = cell(nloc, 1);  end 
if fmeth.lwi,   do_lwi = true;  flwi = cell(nloc, 1); end


disp(['Starting fit for ',num2str(nloc),' pixels.'])
% parfor lops over only the pixels in the mask,
pw = PoolWaitbar(nloc, 'Fitting progress');
for iloc = 1: nloc
    increment(pw) % waitbar
    
    pdata = mdata(iloc,:) ;
    
    
    if do_mono
        fmono{iloc} = fit(TEs', pdata', 'exp1') ;
    end
    
    if do_bi
        fbi{iloc}   = fit(TEs', pdata', 'exp2') ;
    end
    
    if do_lwi
         
        X0this = mX0(iloc,:) ;
        
        fun = @(s) (A*pr(s)) - pdata' ;  
        X = lsqnonlin(fun, X0this, lb, ub, options) ;
        
        A1 = trapz( prt2s, X(1) .*     X(2) .* normpdf(prt2s', X(3), X(5)) ) ;
        A2 = trapz( prt2s, X(1) .* (1-X(2)) .* normpdf(prt2s', X(4), X(6)) ) ;
        LWF = A2 ./ (A1+A2) ;
        
        flwi{iloc} = struct('LWF',LWF, 'X',X) ;
    end
      
end

% Collect outputs
dfit.szmap = szmap ;
dfit.nloc = nloc ;
dfit.loc = loc_mask ;
dfit.fmeth = fmeth ;

if fmeth.mono
    dfit.fmono = fmono ;
end

if fmeth.bi
    dfit.fbi = fbi ;
end

if fmeth.lwi
    dfit.flwi = flwi ;
    dfit.pr = pr ;
    dfit.A = A ;
    dfit.prt2s = prt2s ;
end

