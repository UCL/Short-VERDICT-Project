function cond_diff(grad)
% COND_DIFF Diffusion directions condition number
% 
% cond_diff(grad)
%
% See Batchelor paper

X = grad ;

A = [X.*X 2*X(:,[1,1,2]).*X(:,[2,3,3])];

cond(A)
