function [L, S, rnkL]=RPCA(M,flambda,varargin)
% RPCA Robust Principal Component Analysis
% Wrapping function that calls inexact method
%
% Requires PROPACK libraries, choosvd.m and modification of 
% inexact_alm_rpca.m. Originals all available as a zip file from
%  http://perception.csl.illinois.edu/matrix-rank/Files/inexact_alm_rpca.zip
%
%
% [L S] = RPCA(M)
% [L S] = RPCA(M,flambda)
% [L S] = RPCA(M,flambda, 'param', value, ...)
%
% If flambda is [], it defaults to 1.
% 'param' can be 'maxIter' (defaults to 300),
% 'tol' (defaults to 1e-7)
% 'output', {'none','MLS'}
% 'plotLS', {0 ,1 }
% 
% David Atkinson, adapted from Alex Menys, Pauline Ferry 
%

if exist('choosvd','file') == 0
    disp('Requires choosvd and PROPACK functions.')
    disp(' Try downloading all from: ')
    disp('   http://perception.csl.illinois.edu/matrix-rank/Files/inexact_alm_rpca.zip')
end
    
M = double(M) ;

if nargin == 1
    flambda = 1 ;
elseif nargin > 1 && isempty(flambda)
    flambda = 1 ;
end

% Default values:
maxIter = 300 ;
tol = 1e-7 ;
output = 'none' ;
plotLS = 0 ;

for ip = 1:2:length(varargin)
    switch varargin{ip}
        case {'maxIter', 'MaxIter'}
            maxIter = varargin{ip+1};
        case 'tol'
            tol = varargin{ip+1} ;
        case 'plotLS'
            plotLS = 1 ;
        case 'output'
            output = varargin{ip+1} ;
        otherwise
            warning(['Unrecognised parameter: ',varargin{ip}])
    end
end

ndM = ndims(M) ;

switch ndM
    case 2
        [nv, nt] = size(M) ;
    case 3
        [ny, nx, nt] = size(M) ;
        nv = ny*nx ;
        nz = 1 ;
        M = reshape(M,[nv nt]);
    case 4
        [ny, nx, nz, nt] = size(M) ;
        nv = ny*nx*nz ;
        M = reshape(M,[nv nt]) ;
        
    otherwise
        warning('M should be Casorati or 2D+T or 3D+T')
end

lambda0 = 1/sqrt(max(nv, nt)) ;
lambda = flambda * lambda0 ;

[L, S] = inexact_alm_rpca_da(M, lambda, tol, maxIter, 'plotLS',plotLS);
rnkL = rank(L) ;

if ndM == 3 || ndM == 4
    L = squeeze(reshape(L,[ny nx nz nt])) ;
    S = squeeze(reshape(S,[ny nx nz nt])) ;
end

switch output
    case 'none'
    case 'MLS'
        if ndM == 3 || ndM == 4
            M = squeeze(reshape(M,[ny nx nz nt])) ;
            eshow([M L S])
        end
    otherwise
        warning('Unknown output specified')
end


            
    