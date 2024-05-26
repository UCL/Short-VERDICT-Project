function [Mcurr] = rddr_da(M0, rddr_opts)
% RDDR_DA RDDR Robust Data Driven Registration
% 
% [Mcurr] = rddr_da(M0)
% [Mcurr] = rddr_da(M0, rddr_opts)
%
% M0 is input cine series of size [ny nx nt]
% Mcurr is RDDR output with the same size as M0.
%
% Based on the following paper (please cite):
%  Hamy V, Dikaios N, Punwani S, Melbourne A, Latifoltojar A, 
%    Makanyanga J, Chouhan M, Helbre, E, Menys A, Taylor S, Atkinson D. 
%  "Respiratory motion correction in dynamic MRI using robust data 
%    decomposition registration - Application to DCE-MRI." 
%  Medical Image Analysis 18(2), 301-313 (2013).
%  doi:10.1016/j.media.2013.10.016. 
%
% D.Atkinson@ucl.ac.uk  adapted from code by Valentin Hamy.
% See also imregdemons, RPCA
%

disp('Based on RDDR paper:')
disp('  <a href = "http://dx.doi.org/10.1016/j.media.2013.10.016">Hamy et al. Medical Image Analysis 18(2), 301-313</a>')

[ny, nx, nt] = size(M0) ;
np = nt*ny*nx ;

max_rnk = min([(ny*nx) nt]) ;

if nargin < 2, rddr_opts = [] ; end

if ~isfield(rddr_opts,'start_target_rnk') 
    rddr_opts.start_target_rnk = max_rnk / 4 ;
end
if ~isfield(rddr_opts, 'sps_stop'), rddr_opts.sps_stop = 0.2 ; end
if ~isfield(rddr_opts, 'max_iter'), rddr_opts.max_iter = 10; end
if ~isfield(rddr_opts, 'display_M_per_iter'), rddr_opts.display_M_per_iter = false; end

% Find initial lambda multiplyer based on rank criterion
x0= [0.1 4];
fun = @(x) cfn_flambda0(x,M0,rddr_opts.start_target_rnk) ;
options = optimset('fzero');
options.TolFun = 0.5;
flambda_start = fzero(fun,x0,options) ;

[L0, S0, rnk0] = RPCA(M0, flambda_start) ;
sps0 = nnz(S0)/np ;

sps_curr = sps0 ;
Lcurr = L0 ;
Mcurr = M0 ;
iter = 1 ;
Dtot = zeros([ny nx nt 2]) ;
flambda_vec = flambda_start * log(linspace(exp(1),exp(2.5), rddr_opts.max_iter)) ;

disp(['flambda_start: ',num2str(flambda_start), ... 
    ' rank: ',num2str(rnk0), ' sparsity: ',num2str(sps0)])

while iter <= rddr_opts.max_iter && sps_curr > rddr_opts.sps_stop
    % Register L to frame closest to median, apply deform field to all.
    frame = median_frame(Lcurr) ;
    
    for it = 1:nt
      D = imregdemons(Lcurr(:,:,it),Lcurr(:,:,frame)) ;
      Dtot(:,:,it,:) = Dtot(:,:,it,:) + reshape(D,[ny nx 1 2]);
      Mcurr(:,:,it) = imwarp(M0(:,:,it),squeeze(Dtot(:,:,it,:))) ;
    end
    
    
    % Re-evaluate L,S
    [Lcurr, Scurr, rnkL] = RPCA(Mcurr,flambda_vec(iter)) ;
    
    sps_curr = nnz(Scurr)/np ;
    
    if rddr_opts.display_M_per_iter
        eshow(Mcurr,'Name',['iter: ',num2str(iter), 'median frame: ',num2str(frame)])
    end
    
    disp(['iter: ',num2str(iter),' rnkL: ',num2str(rnkL),' sps_curr: ',num2str(sps_curr)])
    
    iter = iter+1 ;

end % end while loop




function frame = median_frame(data)
% MEDIAN_FRAME
[ny, nx, nt, nd] = size(data) ; %#ok<ASGLU>
if nd ~=1, error(['median_frame expects 2D+t data']), end
Imedian = median(data,3) ;
for it = 1:nt
    frm = data(:,:,it) ;
    dist(it) = norm(Imedian(:) - frm(:)) ;
end
[~,frame] = min(dist) ;



function rnk_diff = cfn_flambda0(x, M, target_rnk )
% CFN_FLAMBDA0

[L, S, rnkL] = RPCA(M,x) ;
rnk_diff = rnkL - target_rnk ;

