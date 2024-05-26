function szinfo(mat)
% SZINFO Size and other information
%
% David Atkinson
% D.Atkinson@ucl.ac.uk
%

% Each addition to str should end with a space.

sz = size(mat) ;
className = class(mat) ;

str = [inputname(1) ': '] ;

str = [str num2str(sz) '. '] ;

if ~isequal(className,'double')
    % double type is default - no need to write out
    str = [ str className ' '] ;
end

if isnumeric(mat)
    if ~isreal(mat)
        str = [ str 'complex '] ;
    end
    str = [str  '(' num2str(min(mat(:))) ' to ' num2str(max(mat(:))) ', median: ' num2str(median(mat(:))) ') '] ;
end

disp(str)