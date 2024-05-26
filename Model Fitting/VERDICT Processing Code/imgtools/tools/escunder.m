function strout = escunder(strin)
% ESCUNDER  Escape the underscore character with a \
%
% strout = escunder(strin)
%
% David Atkinson
% Guy's, King's & St. Thomas' School of Medicine
% @(#)escunder.m	1.1 , created 03/24/01
%

loc = strfind(strin,'_') ;

nunder = length(loc) ;

str = strin;

for iunder = 1: nunder
    str = slash_insert(str,loc(iunder)) ;
  loc = loc + 1 ;
end



strout = str ;


% - - - - - - - - - - - - - - - - - - - - -     
    
function strout = slash_insert(strin, loc)
% strout = slash_insert(strin, loc)

lin = length(strin) ;

strout(1:loc-1) = strin(1:loc-1) ;
strout(loc) = '\' ;
strout(loc+1:lin+1) = strin(loc:lin) ;
