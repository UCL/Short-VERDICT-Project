function [X, Y, Z, POS] = extract_contour(M)
% EXTRACT_CONTOUR Extracts contours from Contour Matrix
%
% [X, Y, Z, POS] = extract_contour(M)
%
% M is a contour matrix from contour or imcontour
% Z is a cell array with the contour heights
% X and Y are cell arrays with each element the x and y coordinates 
% is a cell aray of Position arrays for ROI creation
%
% Example
%  imshow(img,[]), hold on
%  [M,c] = contour(img,[1000 1100 1200]) ;
%  [X, Y, Z, POS] = extract_contour(M) ;
%  plot(X{1}, Y{1})
%
% See also contour ContourProperties


icont = 1 ;
ind = 2 ;
while ind < size(M,2)
    Z{icont} = M(1,ind-1) ;
    N(icont) = M(2,ind-1) ;
    X{icont} = M(1,ind:ind+N(icont)-1) ;
    Y{icont} = M(2,ind:ind+N(icont)-1) ;
    POS{icont} = cat(2,X{icont}', Y{icont}') ;
    
    ind = ind + N(icont) + 1 ;
    icont = icont + 1;
end

