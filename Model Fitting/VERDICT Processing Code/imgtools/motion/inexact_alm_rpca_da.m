function [A_hat E_hat iter] = inexact_alm_rpca_da(D, lambda, tol, maxIter, varargin)
% INEXACT_ALM_RPCA_DA Modified inexact_alm_rpca code for RPCA
%  [A_hat E_hat iter] = inexact_alm_rpca_da(D, lambda, tol, maxIter)
%  ...  = inexact_alm_rpca_da(D, lambda, tol, maxIter, param, value, ...)
%
% DA: Original from http://perception.csl.illinois.edu
%     See Copyright notice below.
%
% param: 'plotLS'  value: plot frequency (0 no plot)
% 
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Robust PCA.
%
% D - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,D-A-E> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   \mu = \rho * \mu;
% end
%
% Based on:
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

% addpath PROPACK;

if nargin < 4
    error(['Call with at least 4 arguments'])
end

fplotLS = 0 ;

if nargin > 4
    np = length(varargin) ;
    for ip = 1:2:np
        switch varargin{ip}
            case 'plotLS'
                fplotLS = varargin{ip+1} ;
        end
    end
end

if fplotLS > 0
    hf = figure('Name','inexact_alm_rpca') ;
    hL = subplot(1,2,1);  ylabel('Rank'), hold on ;
    hS = subplot(1,2,2); , ylabel('Sparsity'), hold on ;
end
    
[m n] = size(D);

% DA defaults removed (passed in from RPCA)

% initialize
Y = D;
if isreal(D)
    norm_two = lansvd(Y, 1, 'L');
else
    singv = svd(Y);
    norm_two = singv(1) ;
end
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = zeros( m, n);
E_hat = zeros( m, n);
mu = 1.25/norm_two ;% this one can be tuned
mu_bar = mu * 1e7;
rho = 1.5         ; % this one can be tuned
d_norm = norm(D, 'fro');

iter = 0;
total_svd = 0;
converged = false;
stopCriterion = 1;
sv = 10;
while ~converged       
    iter = iter + 1;
    
    temp_T = D - A_hat + (1/mu)*Y;
    
    %DA added complex part
    if isreal(D)
        E_hat = max(temp_T - lambda/mu, 0);
        E_hat = E_hat+min(temp_T + lambda/mu, 0);
    else
        E_hat = temp_T ;
        loc = find(abs(E_hat) < lambda/mu) ;
        E_hat(loc) = 0 ;
    end

    
    if choosvd(n, sv) == 1 && isreal(D) 
        [U, S, V] = lansvd(D - E_hat + (1/mu)*Y, sv, 'L');
    else
        [U, S, V] = svd(D - E_hat + (1/mu)*Y, 'econ');
    end
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';    

    total_svd = total_svd + 1;
    
    Z = D - A_hat - E_hat;
    
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    % stop Criterion    
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end    
    
    if fplotLS > 0
       if mod( total_svd, fplotLS) == 0 
           axes(hL), plot(total_svd,rank(A_hat),'ro')
           axes(hS), plot(total_svd, length(find(abs(E_hat)>0)),'ro')
       end
    end
    if mod( total_svd, 10) == 0
        disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' stopCriterion ' num2str(stopCriterion)]);
    end    
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end
