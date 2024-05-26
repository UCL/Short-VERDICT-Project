function explore_xmlroi
% explore_xmlroi  Read ROI saved from Horos as XML file 
% Save ROI from ROI Info panel
%
% Copyright, 2020, David Atkinson
% D.Atkinson@ucl.ac.uk
%
% See also get_OsiriXDBroi
%

ffn = get_data_location('xml_roi','file','gui','always')  ;
dfn = get_data_location('droi','file','gui','always') ;

xmls = parseXML(ffn) ;

nC = length(xmls(2).Children(2).Children)

for iC = 2:2:nC
    xmls(2).Children(2).Children(iC).Name
end

nC = length(xmls(2).Children(2).Children(16).Children) ;

for ipoint = 2:2:nC
    pstr = xmls(2).Children(2).Children(16).Children(ipoint).Children.Data ;
    
    pcell = textscan(pstr,'{%f, %f}') ;
    
    ROIC(ipoint/2,:) = [pcell{1} pcell{2}] ; 
end

dinfo = datparse(dfn);
[vd,md] = d2mat(dinfo,{'slice'},'op','dv') ;
figure
imshow(vd,[]), hold on
roi = images.roi.Polygon(gca,'Position',ROIC)


