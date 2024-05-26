function [fIC, fEES, fVASC, R, rmse, A, t, opt, x] = verdict_fit(scheme,Y,t, opt)
% verdict_fit Perform a VERDICT fit for fIC, fEES, fVASC and R
%
% [fIC, fEES, fVASC, R, rmse]            = verdict_fit(scheme,Y,t, opt)
% [fIC, fEES, fVASC, R, rmse, A, t, opt] = verdict_fit(scheme,Y,t, opt)
% 
% Y [ny nx nz nscheme] or [ny nx nscheme]
% t represent Name-value pairs for tissue properties
%   dEES [2], dIC [2], dVASC [8]
%   Rs   [1:15] (radii)
%
% opt represents name-value pairs
%  mask (0's and 1's) for computation region if speed-up needed, defaults
%  to whole image
%  solver 'SPAMS' | 'lsqnonneg' | {'lsqnonnegTikonhov'} | 'group-lasso-l2'
%
% fIC, fEES, fVASC and R have size of spatial dimensions of Y.
% A [nscheme nr+2]  where nr = length(t.Rs)
%
% Example
%  [fIC, fEES, fVASC, R, rmse] = verdict_fit(scheme,Y,Rs=[5:15], mask=maskvalue)
%
% fIC, fEES, fVASC, R and rmse are only calculated in the mask region. The
% fractional volumes may not sum to 1 due to normalisation problems with 
% DW images.
%
% Solves Ax = y where x are the fractional volumes, y
% is the measured data and A [nscheme nr+2] where nr is the number of
% radii.
% This code does not normalise the fIC, fEES, fVASC outputs so their sum
% may not be exacly 1.
%
% Based in part on VERDICT-AMICO paper: DOI: 10.1002/nbm.4019
%
% Copyright 2022-2023.  David Atkinson
%
% Licence: See licence file in Git repo.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% See also sphereGPD ball astrosticks stejskal 

arguments
    scheme (1,:)  struct
    Y             {mustBeNumeric,mustBeReal}  % [ny nx nmeas] or [ny nx nz nmeas]
    

    t.dEES  (1,1) {mustBeNumeric,mustBeReal} = 2
    t.dIC   (1,1) {mustBeNumeric,mustBeReal} = 2
    t.dVASC (1,1) {mustBeNumeric,mustBeReal} = 8
    t.Rs    (1,:) {mustBeNumeric,mustBeReal} = linspace(0.1,15.1,17) 
%     t.Rs    (1,:) {mustBeNumeric,mustBeReal} = [8,10,12]

    opt.mask {mustBeNumeric,mustBeReal} = []
    opt.solver {mustBeText} = 'lsqnonnegTikhonov'
    opt.ncompart = 2
    opt.NoisyMatrix = false; % If true, average noise magnitude added to matrix elements
    opt.NoiseSigma = 0.05; % Gaussian noise standard deviation (in complex domain ) (1/b=0 SNR)
    opt.lambda1 = sqrt(0.001); % histology regularisation parameter
    opt.lambda2 = sqrt(0.001); % Zeroth order Tikonhov regularisation parameter



end


nr = length(t.Rs) ;
ncompart = opt.ncompart ; % number of compartments beyond radii. 2 for Classic, 1 to exclude vasc
nscheme = length(scheme) ;
szY = size(Y) ;
szmap = szY(1:end-1) ;

if ~exist('opt','var') || isempty(opt.mask)
    opt.mask = ones(szmap) ;
end

if szY(end) ~= nscheme
    warning('MATLAB:verdict_fit:dataSizeInconsistencies', ...
        ['Number of measures (',num2str(szY(end)), ...
        ') must be same as size of schemes (', num2str(nscheme),')'])
end

solver = opt.solver ;
if isequal(solver,'SPAMS')
    if ~exist('mexLasso')
        warning('MATLAB:verdict_fit:solverMex', ...
            'Solver SPAMS needs mexLasso')
    end
end


% Number of unknowns
nUnknown = nr + ncompart;

%% Construct Matrix

A = zeros([nscheme nr+ncompart]) ; 
sIC  = zeros([nscheme nr]) ;
sEES = zeros([1 nscheme]) ;
sVASC = zeros([1 nscheme]) ;

for ischeme = 1:nscheme

    % == Sphere signals
    for ir = 1:nr
        rsignal = sphereGPD(scheme(ischeme).delta, scheme(ischeme).DELTA, ...
           scheme(ischeme).G, t.Rs(ir), t.dIC);

        % Add noise average
        if opt.NoisyMatrix  && (scheme(ischeme).bval ~= 0)
            rsignal = RicianNoiseAverage(rsignal, opt.NoiseSigma);
        end

        sIC(ischeme,ir) = rsignal;
    end


    % == EES signal
    EESsignal = ball(scheme(ischeme).bval, t.dEES) ;
    % Add noise average
    if (opt.NoisyMatrix) && (scheme(ischeme).bval ~= 0)
        disp(EESsignal)
        EESsignal = RicianNoiseAverage(EESsignal, opt.NoiseSigma);
        disp(EESsignal)
    end
    sEES(ischeme)  = EESsignal;


    % == VASC signal
    VASCsignal = astrosticks(scheme(ischeme).bval, t.dVASC) ;
    % Add noise average
    if opt.NoisyMatrix && (scheme(ischeme).bval ~= 0)
        VASCsignal = RicianNoiseAverage(VASCsignal, opt.NoiseSigma);
    end
    sVASC(ischeme)  = VASCsignal;


    if ncompart == 2
        A(ischeme,:)   = [sIC(ischeme,:) sEES(ischeme) sVASC(ischeme) ] ;
    else
        A(ischeme,:)   = [sIC(ischeme,:) sEES(ischeme)] ;
    end
end



%% Fitting

% fRs = zeros(szmap, length(t.Rs)); % Radii volume fractions
fIC   = zeros(szmap) ; % fIC map
fEES  = zeros(szmap) ;
fVASC = zeros(szmap) ;
R     = zeros(szmap) ; % radius map
rmse = zeros(szmap) ; % fit rmse

opt.mask = opt.mask(:) ;
fIC = fIC(:); fEES = fEES(:); fVASC = fVASC(:); R = R(:) ;
Y = reshape(Y,[prod(szmap) nscheme]) ;



locinmask = find(opt.mask) ;

for ip = 1:length(locinmask)  % can be parallelised

    ind = locinmask(ip) ;
    y = Y(ind,:) ;

    percent_done = (ip/length(locinmask))*100
    
    % Remove infinties and NaN
    y(isinf(y)) = 0;
    y(isnan(y)) = 0;

    % Y = A x
    % Solving with x = lsqnonneg(A, y(:))
    %  is similar to x = mldivide(A,y(:))  or A\y(:)


    % The lasso method below is slow and this lambda doesn't produce nice fIC
    %     yv = y(:) ;
    %     yv(~isfinite(yv))=0 ;
    %     B = lasso(A, yv, 'Lambda',0.001) ;
    %     x = B(:,1) ;

    switch solver

%         case 'lsqlin_well_determined'
% 
%             [m,n] = size(A);
%        
% 
%             % == Constraint sum(y) = 1
%             Aeq = zeros(n,n);
%             Aeq(1,:) = 1.0;
% 
%             beq = zeros(n,1);
%             beq(1) = 1;
% 
%             x = lsqlin(A, y(:), [],[], Aeq, beq, zeros(n,1), ones(n,1));
% 
%         case 'lsqlin_underdetermined'
% 
%             nUnknown = size(A,2) ;
%             lambda = sqrt(1e-3) ;
%             L = eye(nUnknown) ;
% 
%             AT = [A; lambda*L] ;
%             YT = [y(:) ; zeros([nUnknown 1])] ;
% 
%             [m,n] = size(A);
%     
%             % == Constraint sum(y) = 1
%             Aeq = zeros(n,n);
%             Aeq(1,:) = 1.0;
% 
%             beq = zeros(n,1);
%             beq(1) = 1;
% 
%             x = lsqlin(AT, YT(:), [],[], Aeq, beq, zeros(n,1), ones(n,1));


        case 'histreg'

            %% Regularisation based on histology prior distribution

            % ===== Construct histology distribtuion

            % Assume Gaussian distribution
            mu = 8;
            sigma = 2;
            HistDist = normpdf(t.Rs, mu, sigma);
            % Normalise! 
            HistDistMag = sum(HistDist);
            HistDist = (1/HistDistMag)*HistDist;
           
            % ===== Construct regularisation functional for histology prior

%             % == First formulation (dot product)
% 
%             L1 = zeros(nUnknown);
%             L1(1, 1:nr) = (HistDist);
% 
%             % Set regularisation parameter 
%             lambda1 = 0.1;
% 
%             % Construct target 
%             histtarget = zeros(nUnknown,1);
%             histtarget(1) = 1;

            % == Second formulation (squared residuals)

            L1 = eye(nUnknown);
            L1(end, end) = 0; % fEES row

            % Regularisation parameter
            lambda1 = opt.lambda1;

            % Construct target
            histtarget = zeros(nUnknown,1);
            histtarget(1:end-1) = HistDist;

            

            % Construct regularisation function for zeroth order Tikonhov
            L2 = eye(nUnknown);
            
% %             Construct regularisation function for first order Tikonhov]
% %             (Finite difference matrix)
% 
%             L2 = zeros(nUnknown);
%             L2(1:nr,1:nr) = -2*eye(nr);
%             for rowindx = 1:nr
%                 L2(rowindx, rowindx+1) = 1;
%                 L2(rowindx+1, rowindx) = 1;
%             end
%             L2(nr,1)=1;
%             L2(1,nr) = 1;

            lambda2 = opt.lambda2;
            

            % Construct augmented matrix
            AT = [A; lambda1*L1; lambda2*L2] ;
            YT = [y(:) ; lambda1*histtarget; zeros([nUnknown 1])] ;

            % Solve
            x = lsqnonneg(AT, YT) ;





        case 'lsqnonneg'
            x = lsqnonneg(A, y(:)) ;

        case 'lsqnonnegTikhonov'
            nUnknown = size(A,2) ;
            lambda =sqrt(0.001) ;


            L = eye(nUnknown) ;

            AT = [A; lambda*L] ;
            YT = [y(:) ; zeros([nUnknown 1])] ;

            x = lsqnonneg(AT, YT) ;
        case 'SPAMS'
            % See https://github.com/daducci/AMICO_matlab/blob/master/models/AMICO_VERDICTPROSTATE.m
            % for SPAMS mexLasso call.
            SPAMS_param.mode    = 2;
            SPAMS_param.pos     = true;
            SPAMS_param.lambda  = 0;    % l1 regularization
            SPAMS_param.lambda2 = 1e-3; % l2 regularization

            norms = repmat( 1./sqrt(sum(A.^2)), [size(A,1),1] );
            Anorm = A .* norms;
            xnorm = full( mexLasso( y(:), Anorm, SPAMS_param ) );
            x = xnorm .* norms(1,:)';
        case 'group-lasso-l2'
            % adapted from AMICO_VERDICTPROSTATE
            % Seems to be too slow.
            SPAMS_param.mode    = 2;
            SPAMS_param.pos     = true;
            SPAMS_param.lambda  = 0;    % l1 regularization
            SPAMS_param.lambda2 = 1e-3; % l2 regularization
            SPAMS_param.loss = 'square' ;
            SPAMS_param.regul = 'group-lasso-l2' ;
            SPAMS_param.groups = int32([ repmat(1,1,nr) 2 3 ]);  % all the groups are of size 2
            SPAMS_param.intercept = false ;
            x = full( mexFistaFlat( y(:), A, zeros(size(A,2),1), SPAMS_param ) );

        otherwise
            error(['unknown solver: ',solver])
    end

    fIC(ind) = sum(x(1:nr)) ;
    R(ind)   = sum(x(1:nr)'.*t.Rs) / fIC(ind) ;
    fEES(ind) = x(nr+1) ;

    if ncompart ==2
        fVASC(ind) = x(nr+2) ;
    end

    rmse(ind) = norm(y(:) - A*x) / sqrt(nscheme) ;
end

fIC = reshape(fIC,szmap) ;
fEES = reshape(fEES,szmap) ;
fVASC = reshape(fVASC,szmap) ;
R = reshape(R,szmap) ;
rmse = reshape(rmse, szmap) ;

end
