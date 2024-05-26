function roi = osirix2val(filename)
% OSIRIX2VAL Extract pixel values from Osirix ROIs saved as XML file
%
% roi = osirix2val(filename)
% roi = osirix2val
%
% roi has fields name and vals
%
% For example to get a histogram of the 2nd roi
%  roi = osirix2val ;
%  hist(roi(2).vals)
%
%
% D.Atkinson@ucl.ac.uk
% See also xml2struct_28518

if nargin < 1 || ~exist(filename,'file')
   [fn,pn] = uigetfile('*.xml') ;
   filename = fullfile(pn,fn) ;
end

% xml2struct_28518 is from FileExchange and modified
xml = xml2struct_28518(filename) ;

nroi = length(xml.plist{2}.dict.array.dict) ;

for iroi = 1:nroi
    if nroi==1
        struc = xml.plist{2}.dict.array.dict ;
    else
        struc = xml.plist{2}.dict.array.dict{iroi} ;
    end
    
    nm = struc.array.dict.string{2}.Text ;
    nv = length(struc.array.dict.array{3}.real); 

    vals = zeros([1 nv]) ;

    for iv = 1: nv
        vals(1,iv) = str2double(struc.array.dict.array{3}.real{iv}.Text) ;
    end
    
    roi(iroi).vals = vals ;
    roi(iroi).name = nm ;
  
    disp([nm, ': Mean ',num2str(mean(vals)),', std ',num2str(std(vals))])
end


