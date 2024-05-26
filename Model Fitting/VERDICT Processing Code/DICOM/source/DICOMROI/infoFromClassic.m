function refInfo = infoFromClassic(fns)
% INFORFROMCLASSIC DICOM meta data info array from Classic single-frame
% filenames
%
% refInfo = infoFromClassic(fns)
% fns is a cell array of filenames, typically from dselector
%
% Copyright 2021. David Atkinson
%
% See also dicomRTStruct  dselector dselect

for ifile = 1:length(fns)
    refInfo(ifile) = dicominfo(fns{ifile}) ;
end