function outCoords= convData(inCoords, inData, outData)
% convData Convert coordinates between XData and YData scales
%  outCoords= convData(inCoords, inData, outData)
%
% D.Atkinson@ucl.ac.uk
%


outCoords = outData(1) + ...
    ( inCoords-inData(1) ) / (inData(2)-inData(1)) * (outData(2)-outData(1)) ;

