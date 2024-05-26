function [varargout] = MFAT1(M, FAs, TRs, B1, varargin)
% MFAT1 Multi Flip Angle T1.
%   Calculates a T1 value for multi flip angle data by fitting the
%   linearised or non-linear Ernst function to measured data.
%
%   [fit] = MFAT1(M, FAs, TRs, B1, 'calc_nl', value)
%   [fit] = MFAT1(M, FAs, TRs, B1, 'param', value, ...)
% 
% Variables
%   M   - [ np , nfa]  or [ny nx nfa] or [ ny nx nz nfa]
%   FAs - Flip Angle (degrees) [ 1 nfa]
%   TRs - (ms) scalar [1] or vector with one TR per Flip Angle  [1 nfa]
%   B1  - (B1-derived flip angle correction (%)) [ size of M without last dim ]
%           B1 can be [], in which case it defaults to 100%
%
% parameters (default values displayed in {})
%   'thresh'    - Threshold for pixels excluded [useage Mmin=max(M)/thresh]
%               - {15}
%   'x0_meth'   - Method for determining non-linear initialisation
%               - {'linear'}, 'zero'
%   'calc_nl'   - determines if the non-linear Ernst function will be used.
%               - {true}, false
%   'B1unknown' - determines if B1 should be fit. Requires more than one TR.
%               - {false}, true   
%   'optdisplay'- determines level of output for lsqnonlin function.
%               - {'off'}, 'on' , 'iter'
%   'B1medfilt' - 2-element vector required if performing 2-D median filter
%                   on B1 values using medfilt2().
%
% fit is a structure with possible fields:
%   T1_lin, T1_nl, M0_lin, M0_nl, resn_lin, resn_nl
%
% Example usage:
%   dinfo = datparse ;  % select top folder for MFA single-frame
%   [volip, matip] = d2mat(dinfo,{'slice','fa','wfio'},'wfio',3,'op','fp') ;
% OR for source DIXON
%   [volip, matip] = d2mat(dinfo,{'slice','fa','echo'},'echo',1,'op','fp') ;
%   db1 = datparse ;  % select B1 mapping scan
%   [vb1,matb1] = d2mat(db1,{'slice'},'op','fp') ;
% OR MultiFrame
%   db1 = datparse(dselect) ;  % select B1 map sequence
%   [vb1,matb1] = d2mat(db1,{'slice','itype'},'itype',7,'op','fp') ; %
%   (type 7 for dual angle method)
%   % Group multiple flip angles into one folder, and select all:
%   dfa = datparse(dselect) ; % select ALL multi flip angle scans
%   [volip, matip] = d2mat(dfa,{'slice','fa','echo'},'echo',1,'op','fp') ;
%   vb1r = dreslice(vb1,matb1,matip) ;
%   slc=42; % slice number you require
%   fit = MFAT1(volip(:,:,slc,:),matip.faVec,matip.RepetitionTime_Vec,vb1r(:,:,slc),'calc_nl' ,false) ;
%   T1display(fit.T1_lin)
%
% OR (single frame):
%   [vb1,matb1] = d2mat(dinfo,{'slice', 'itype','series'},'itype',10,'series',3301,'op','dv') ;
%   vb1r = dreslice(vb1,matb1,matip) ;
%   fit = MFAT1(volip(:,:,75,:),matip.faVec,matip.RepetitionTime_Vec,vb1r(:,:,75),'calc_nl' ,false) ;
%   T1display(fit.T1_lin)
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also LLT1 T1DISPLAY D2MAT DATPARSE LSQNONLIN MEDFILT2

% Identify deprecated output behaviour 
if nargout>1
    % two output arguments could result in non-linear calculations being
    % performed without allowing the results to output.
    error('imgtools:MFAT1:outputArgumentCheck',['MFAT1 too many output ',...
        'arguments. Please provide an output structure.'])
end

% Defaults for user defineable parameters
thresh = 15 ;           % threshold for T1 calc
calc_nl = true ;        % switch on non-linear calculation
x0_meth = 'linear' ;    % method for initialising non-linear calculation
optdisplay = 'off';     % lsqnonlin 'Display' options
B1unknown = false ;     % calculate B1
filtB1 = false ;        % filter B1

% not input parameters
B1min = 0.01 ;          % B1 should be over 1%
calc_grid = false ;     
clipT1 = true ;         % clips linear T1s to be between 0 and T1max
T1max = 10000 ;
parUserDefined=false;   % gives warning if calc_nl not set by user

% parse input parameters
for ivar = 1:2:length(varargin)
    switch varargin{ivar}
        case {'threshold', 'thresh'}
            thresh = varargin{ivar+1} ;
        case {'x0', 'x0_meth'}
            x0_meth = varargin{ivar+1} ;
        case 'calc_nl'
            calc_nl = varargin{ivar+1};
            parUserDefined=true;
        case 'optdisplay'
            optdisplay = varargin{ivar+1} ;
        case 'B1unknown'
            B1unknown = varargin{ivar+1} ;
        case 'B1medfilt'
            filtB1 = true;
            B1medfilt = varargin{ivar+1} ;
            if length(B1medfilt) ~= 2
                warning(['B1medfilt should be 2-element vector e.g.[30 30]'])
            end
        otherwise
            warning(['Parameter not recognised: ',varargin{ivar}])
    end
end

% warn of long calculation if non-linear not specifically requested
if ~parUserDefined
    warning('imgtools:MFAT1:calcMethodUnset',['calc_nl was not set by',...
        ' user. MFAT1 defaulting to non-linear calculations. This may',...
        'result in long calculations if nl values are not required.'])
end

% check for 

M = double(M) ; FAs = double(FAs) ;

szM = size(M) ;
if ndims(M) == 2
    szOP = [1 szM(1)] ;
else
    szOP = szM(1:(ndims(M)-1));
end

ndM = ndims(M) ;
npix = prod(szM(1:ndM-1)) ;

if nargin > 3 && ~isempty(B1)
    B1 = double(B1)/100 ;
else
    B1 = ones(szOP) ;
end

if filtB1 % median filter B1
    for islice = 1:size(B1,3)
        B1(:,:,islice) = medfilt2(B1(:,:,islice),B1medfilt) ;
    end
end

B1 = B1(:) ;

if length(B1) ~= npix
    error(['B1 has different number of pixels to input data'])
end

utr = unique(TRs) ;
if length(utr) == 1 && B1unknown
    error(['For unknown B1, need at least two TRs'])
end

% check FAs size, convert to radians
FAs = FAs * 2 * pi / 360 ;

nfa = szM(ndM) ; % last dimension of M must be flip angle
szFAs = size(FAs) ;


nFAs = length(FAs) ;

if nFAs ~= nfa
    error(['Number of flip angles do not correspond.'])
end


if length(TRs) == 1
    TRs = repmat(TRs,[1 nfa]) ;
else
    if length(TRs)~=nfa
        error(['Vector TRs needs to be same as number of FAs.'])
    end
    TRs = TRs(:)' ; % make row
end

%
M = reshape(M,[npix nfa]) ;

% Linear fit to Ernst function.
%%%%% Linear regression (Fram's method) %%%%%%
% eqn from Wang 2010
% SI(theta)/sin(theta) = E * SI(theta)/tan(theta) + M*(1-E)
%           y          = m *          x           +    c
%
% ! Cannot do AX=B trick with B having more than 1 column because A depends
% on the signal

tr = unique(TRs) ;
if length(tr) > 1
    warning(['Linear fit must have one TR, using mean.'])
    tr_lin = mean(tr) ;
else
    tr_lin = tr ;
end

T1_lin = zeros([npix 1]) ;
M0_lin = zeros([npix 1]) ;
resn_lin = zeros([npix 1]) ;

Msum = sum(M,2) ; % summed along FA
loc = find(Msum > max(Msum)/thresh  & B1 > B1min) ;

disp(['Linear fitting T1s from ',num2str(length(loc)),' pixels.']) ;

nloc = length(loc) ;
wintv = round(nloc/100) ;

T1_par = zeros([nloc 1]) ;
M0_par = zeros([nloc 1]) ;
resn_par = zeros([nloc 1]) ;

% Allow for parallel operation
% Assumes preferences set so that parpool opens automatically

if exist('parpool')
       
    M_broadcast=M(loc,:);
    B1_broadcast=B1(loc);
    
    parfor il = 1:nloc
        
        y = M_broadcast(il,:)./sin(B1_broadcast(il)*FAs) ;
        x = M_broadcast(il,:)./tan(B1_broadcast(il)*FAs) ;
        [soln,S] = polyfit(x,y,1) ;
        m = soln(1) ;
        c = soln(2) ;
        
        % In parfor array slicing prevents using a different indexing from loop var
        
        T1_par(il) = -tr_lin/log(m) ;
        M0_par(il) = c/(1-m) ;
        resn_par(il) = S.normr ;
        
    end
    
else
    hw = waitbar(0,['Linear fitting T1s from ',num2str(length(loc)),' pixels.']) ;
    for il = 1:nloc
        if rem(il,wintv) == 0
            waitbar(il/nloc,hw) ; % too many drawnows can slow computation
        end
        
        ipix = loc(il) ;
        y = M(ipix,:)./sin(B1(ipix)*FAs) ;
        x = M(ipix,:)./tan(B1(ipix)*FAs) ;
        [soln,S] = polyfit(x,y,1) ;
        m = soln(1) ;
        c = soln(2) ;
        
        T1_par(il) = -tr_lin/log(m) ;
        M0_par(il) = c/(1-m) ;
        resn_par(il) = S.normr ;
    end
    close(hw) ;
end

T1_lin(loc) = T1_par ;
M0_lin(loc) = M0_par ;
resn_lin(loc) = resn_par ;


if clipT1
    loc_imag = find( abs(imag(T1_lin)) > 0 );
    loc_bad = find(T1_lin < 0 | T1_lin > T1max) ;
    loc_bad = [loc_bad ; loc_imag] ;
    disp('clipping T1 values out of bounds in T1_lin.');
    disp([num2str(length(loc_bad)),' locations replaced.']);   
    T1_lin(loc_bad) = 0 ;
    M0_lin(loc_bad) = 0 ;
end

if calc_grid
    T1s = [300:50:2000];
    B1s = [0.5:0.01:1.5] ;
    M0s = [0.5:0.01:1.5];
    
    mincf = realmax ;
    for iT1 = 1:length(T1s)
        for iB1 = 1:length(B1s)
            for iM0 = 1:length(M0s)
                x(1) = T1s(iT1);
                x(2) = M0s(iM0) * 30 ;
                x(3) = B1s(iB1) ;
                
                cf = ernstcf(x,FAs,TRs, M) ;
                ncf = norm(cf) ;
                if ncf < mincf
                    mincf = ncf ;
                    T1opt = T1s(iT1);
                    M0opt = M0s(iM0) ;
                    B1opt = B1s(iB1) ;
                end
            end
        end
    end
    T1opt
    M0opt
    B1opt
    
end


% Non-linear
if calc_nl
    T1_nl = zeros([npix 1]) ;
    M0_nl = zeros([npix 1]) ;
    if B1unknown
        B1_nl = zeros([npix 1]) ;
    end
    resn_nl = zeros([npix 1]) ;
    
    options = optimset('Display',optdisplay) ;
    hw = waitbar(0,['Non-lin fitting T1s from ',num2str(length(loc)),' pixels.']) ;
    if exist('parpool')
        
        D = parallel.pool.DataQueue;
        afterEach(D, @nUpdateWaitbar); % allows waitbar use in parfor
        p=1;
        
        T1_nlpar = zeros([nloc 1]) ;
        M0_nlpar = zeros([nloc 1]) ;
        resn_nlpar = zeros([nloc 1]) ;
        
        if B1unknown
            B1_nlpar = zeros([nloc 1]) ;
        end
        
        
        M_broadcast=M(loc,:);
        B1_broadcast=B1(loc);
        T1_broadcast=T1_lin(loc);
        M0_broadcast=M0_lin(loc);
        
        parfor il = 1:nloc
                        
            x0 = x0parasign_bc(x0_meth,T1_broadcast(il),M0_broadcast(il),B1unknown);
            
            fn = @(x)ernstcf(x, B1_broadcast(il)*FAs, TRs, M_broadcast(il,:)) ;
            
            [x,resnorm,~,~] = lsqnonlin(fn,x0,[],[],options) ;
            T1_nlpar(il) = x(1) ;
            M0_nlpar(il) = x(2) ;
            if B1unknown
                B1_nlpar(il) = x(3) ;
            end
            
            resn_nlpar(il) = resnorm ;
            send(D,il) % triggers nUpdateWaitbar through afterEach
        end
        T1_nl(loc)=T1_nlpar;
        M0_nl(loc)=M0_nlpar;
        
        if B1unknown
            B1_nl(loc) = B1_nlpar;
        end
        
        resn_nl(loc) = resn_nlpar;
    else
        for il = 1:nloc
            if rem(il,wintv) == 0
                waitbar(il/nloc,hw) ; % too many drawnows can slow computation
            end
            ipix = loc(il) ;
            
            switch x0_meth
                case 'linear'
                    x0(1) = T1_lin(ipix)  ;
                    x0(2) = M0_lin(ipix)  ;
                case 'zero'
                    x0(1) = 0  ;
                    x0(2) = 0  ;
                case 'guess1'
                    x0(1) = 1500 ;
                    x0(2) = 30 ;
                otherwise
                    error(['Unknown starting estimate method'])
            end
            
            if B1unknown
                x0(3) = 1 ;
            end
            
            fn = @(x)ernstcf(x, B1(ipix)*FAs, TRs, M(ipix,:)) ;
            
            [x,resnorm,residual,exitflag] = lsqnonlin(fn,x0,[],[],options) ;
            T1_nl(ipix) = x(1) ;
            M0_nl(ipix) = x(2) ;
            if B1unknown
                B1_nl(ipix) = x(3) ;
            end
            
            resn_nl(ipix) = resnorm ;
        end
    end
    close(hw) ;
    
    
    if clipT1
        % revert to linear fit when T1 out of bounds
        loc_imag = find( abs(imag(T1_nl)) > 0 );
        loc_bad = find(T1_nl < 0 | T1_nl > T1max) ;
        loc_bad = [loc_bad ; loc_imag] ;
        disp('clipping T1 values out of bounds in T1_nl.');
        disp([num2str(length(loc_bad)),' locations replaced.']);
        T1_nl(loc_bad) = T1_lin(loc_bad) ;
        M0_nl(loc_bad) = M0_lin(loc_bad) ;
    end
    
    resn_nl = reshape(resn_nl, szOP) ;
    T1_nl = reshape(T1_nl, szOP) ;
    M0_nl = reshape(M0_nl, szOP ) ;
    resn_nl = reshape(resn_nl, szOP) ;
    if B1unknown
        B1_nl = reshape(B1_nl, szOP) ;
    end
    
end

T1_lin = reshape(T1_lin,szOP) ;
M0_lin = reshape(M0_lin, szOP) ;
resn_lin = reshape(resn_lin, szOP) ;

if nargout == 2
    varargout{1} = T1_lin ;
    varargout{2} = M0_lin ;
else
    fit.T1_lin = T1_lin;
    fit.M0_lin = M0_lin ;
    fit.resn_lin = resn_lin ;
    if calc_nl
        fit.T1_nl = T1_nl;
        fit.M0_nl = M0_nl ;
        fit.resn_nl = resn_nl ;
        if B1unknown
            fit.B1_nl = B1_nl ;
        end
    end
    varargout{1} = fit ;
    
end


    function nUpdateWaitbar(~)
        % nested function allows parfor waitbar utility. Needs to be able
        % to see p, nloc, hw and wintv.
        
        if ~mod(p,wintv)
            waitbar(p/nloc, hw);
        end
        p = p + 1;
    end

end


function cf = ernstcf(x,FAs_rad,TR, SI)
% Non-linear fit to Ernst Function
% SI = M .* sin(FAs_rad).*((1-E)./(1-cos(FAs_rad).*E));
% x(1) = t1
% x(2) = M0
% x(3) = B1 (optional)

if length(x)> 2
    B1 = x(3) ;
    FAs_rad = FAs_rad * B1 ;
end

TR = TR(:)' ;

E = exp(-TR/x(1)) ;

cf = (x(2) .* sin(FAs_rad) .*((1-E)./(1-cos(FAs_rad).*E)))  - SI ;
end




function x0=x0parasign_bc(x0_meth,T1_val,M0_val,B1unknown)
% Asigning values to individual vector entries confuses matlabs variable
% classification in parfor loops. Moving asignment to a separate function
% solves the problem.
% Modified for optimised broadcast variables

switch x0_meth
    case 'linear'
        x0(1) = T1_val;
        x0(2) = M0_val;
    case 'zero'
        x0(1) = 0  ;
        x0(2) = 0  ;
    case 'guess1'
        x0(1) = 1500 ;
        x0(2) = 30;
    otherwise
        error(['Unknown starting estimate method'])
end

if B1unknown
    x0(3) = 1 ;
end
end
