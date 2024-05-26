function y= msysmatv(x, transp_flag, opt)
% MSYSMATV Pre-conditioning 
%
%
% David Atkinson.  D.Atkinson@ucl.ac.uk
%
% See also SIMUL
%

if strcmp(transp_flag,'notransp') % 'notransp') returns M\x
    % If FWD model A is m x n, then M should be n x n. So, n must
    % be image size and input x must be image
    % W = reshape(x,[nfe npe_f]) ;
    
    W = x ;
    cw = opt.pcsos .* W ;
    
    y = cw(:) ;
    
elseif strcmp(transp_flag,'transp') % returns M'\x.
    % W = reshape(x,[nfe npe_f]) ;
    W = x ;
    cw = conj(opt.pcsos) .* W ; % transpose is of full diag matrix. pcsos is square.
    
    y = cw(:) ;
end
end

