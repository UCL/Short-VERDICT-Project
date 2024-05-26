function [A, count] = freadbuf(t, prec)
% FREADBUF FREAD from buffer
%
% [A, count] = freadbuf(t, prec)
%  t is tcpip object
%  prec is precision e.g. 'float'
%
% Inserts 0.1s pauses - may need to be adjusted in some circumsatnces
%
% D.Atkinson@ucl.ac.uk
%

count = 0 ;
A = [] ;

% set bytes per value
switch prec
    case {'uint16','int16'}
        bypv = 2 ;
    case {'float','float32'}
        bypv = 4 ;
    otherwise
        warning(['Precision ',prec,' mot implemented.'])
end
        

while t.BytesAvailable > bypv-1  
    [bA, bcount] = fread(t, t.BytesAvailable/bypv, prec ) ;
    
    count = count + bcount ;
    A = [A ; bA] ;
    pause(0.1)
end