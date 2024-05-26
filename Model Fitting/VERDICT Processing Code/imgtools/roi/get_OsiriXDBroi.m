function [points, dinfo] = get_OsiriXDBroi(messg)
% get_OsiriXDBroi Get ROI from OsiriX database.
% In development, but benefit will be that user can draw ROIs using full
% power of OsiriX and no need to specifically save/name.
% 
%
%  [points, dinfo] = get_OsiriXDBroi(messg)
%
% User selects DICOM SR ROI from OsiriX databae. Returns points on ROI and
% the dinfo structure.
%
% Currently only for line and close polygon ROIs.
% ! Interpretation of Encapsulated Document to get points is a hack and 
%   may break.
%
% ! Slice order may differ after DICOM is read using d2mat
%
% ! In OsiriX, the polygon has curved connections between points.
%
% Expected Workflow
%  1) Tidy up OsiriX
%  2) Read in DICOMs 
%  3) Draw ROIs as normal on one type, e.g. T2W.
%  4) Click to another image (to force save of ROIs)
%  5) Load the same DICOM (e.g. T2W) into eshow using d2mat first
%  6) Position eshow slice on slice for ROI
%  7) Press eshow butting that calls this function.
%
% Copyright, 2020, David Atkinson, University College London.
% D.Atkinson@ucl.ac.uk
%
% See also eshow dfastinfo dselect
%

if nargin ==0
    messg = 'Select SR ROI';
end

dfn = get_data_location('get_OsiriXDBroi','dselect','gui','always', ...
    'dselectpvp',  {'message', messg} )  ;

if isempty(dfn)
    points=[]; dinfo=[];
    return
end

dinfo = dfastinfo(dfn{1}) ;

ed = [char(dinfo.EncapsulatedDocument)'] ;

for ic = 1:length(ed)
    disp([ed(ic),'   ',num2str(dinfo.EncapsulatedDocument(ic))])
end


% Following section would be better if it searched for {num, num} patterns.
% Liable to break. Would be better if data could be correctly deserialized.

% Polygon ROI points start'({' ?
%  Note sure, the character before the { might just be the string length
%  converted to a char
%   Need to check for Closed Polygon, Rectangle and Oval?

stbp = strfind(ed,'{{') ; % start of brace pair
enbp = strfind(ed,'}}') ; % end of brace pair

nbp = length(stbp) ;
if length(enbp) ~= nbp
    warning('Unequal number of brace pairs')
else
    disp([num2str(nbp), ' brace pairs found.'])
end

for ibp = 1:nbp
    disp(['ROI: ',num2str(ibp),' type ',ed(stbp(ibp)-1) ])
    
    useful = ed(stbp(ibp)+1:enbp(ibp)) ;
    sc = textscan(useful,'%f','Delimiter',{'{','}',','}, 'MultipleDelimsAsOne',true)  ;
    
    P = sc{1};
    points = cat(2,P(1:2:end), P(2:2:end)) 

end

    


stp = strfind(ed,'({') ;
st2 = strfind(ed,'''{') ;

stp = sort([stp st2]) ;

if isempty(stp)
    % rectangle?

    st = strfind(ed,'{{') ;
    fin = strfind(ed, '}}') ;
    
    useful = ed(st+1:fin) ;
    
    sc = textscan(useful,'%f','Delimiter',{'{','}',','}, 'MultipleDelimsAsOne',true) ;
    
    P = sc{1};
    points = cat(2,P(1:2:end), P(2:2:end)) ;
else
    points = zeros([length(stp) 2]) ;
    for istp = 1:length(stp)
        useful = ed(stp(istp)+1  :end) ;
        fb = strfind(useful,'}') ;
        useful = useful(1:fb(1)) ;
        
        sc = textscan(useful,'%f','Delimiter',{'{','}',','}, 'MultipleDelimsAsOne',true) ;
        points(istp,:) = [sc{1}(1)  sc{1}(2)] ;
    end
end






