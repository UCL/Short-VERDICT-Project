function col_order = colorder(ncol)
% COLORDER Set color order for lines in plots.
% col_order = colorder(ncol)
% 
% For ncol < 7 uses colours similar to MATLAB
% For ncol >= 7, 
%   blue
%   light red
%   green
%   darkening shades of gray
%   ...
%   purple  (last line)
%
% Author: David Atkinson  D.Atkinson@ucl.ac.uk

% Copyright UCL 2014.
%

% colorder = [ ...
%         0  0     1 ;
%        0  1     0 ;
%        0  1     1 ;
%         1  0     1 ;
%         1  1     0 ;
%         1  0.5  0.25 ] ;
    
    
if ncol < 7
  col_order = [  0    0     1.0 ;
                 0    0.5    0  ;
                 1.0   0      0 ;
   0       0.75   0.75 ;
   0.75                  0   0.75 ;
   0.75   0.75                  0 ;
   0.25   0.25   0.25  ] ;
else
  gmax = 0.75 ;
  ngray = ncol - 4 ;
  m = (-gmax)/(ngray-1) ;
  c = gmax-m ;
  col_order = zeros(ncol,3) ;
  col_order(1,:) = [ 0 0   1] ;
  col_order(2,:) = [ 0 1   0] ;
  col_order(3,:) = [ 1 0   0] ;
  
  
  for igray = 1:ngray-1
    col_order(3+igray,:) = [1 1 1]*(m*igray + c) ;
  end
  col_order(end-1,:) = [ 0 0 0] ; % stops rounding error
  col_order(end,:) = [0 1 1] ;
  
end


