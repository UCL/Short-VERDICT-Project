function dinfo = xmlparse(xmlparfn)
% XMLPARSE Parse Philips XML files
%  dinfo = xmlparse
%  dinfo = xmlparse(xmlparfn) 
%
%  Returns dinfo empty if there is an error parsing XML file.
%
% Intended to be used with d2mat. Mirrors dparse for DICOM files.
% 
%
% David Atkinson
%
% See also DPARSE D2MAT XML2STRUCT DMFPARSE

if nargin==0 || ~exist(xmlparfn,'file')
  global XMLFN   % ensures name persist across calls to this 
                % function (for ease of use and testing, not essential)
                        
  [fn,pn] = uigetfile({'*.xml', '*.XML'}, 'Select Philips XML file.', XMLFN) ;
  if isequal(fn, 0) ; dinfo = [] ; return; end ;
  XMLFN = fullfile(pn,fn) ;
 
  xmlparfn = XMLFN ;
end

disp(['Parsing xml file: ',xmlparfn])
out = xml2struct(xmlparfn) ;

[pstr,nm,ext]=  fileparts(xmlparfn) ;
switch ext
 case '.XML'
  recfn = [fullfile(pstr,nm),'.REC'] ;
 case '.xml'
  recfn = [fullfile(pstr,nm),'.rec'] ;
end

SERIES = 2 ;
IMARR = 4 ;
KEY = 2 ;

bpv = 2 ;

RL = 1 ; AP = 2 ; FH = 3 ; % used for reading-in diffusion directions

sitems = out.children(SERIES).children ;

% show the Series information
% show_block(sitems) ;
bout = read_block(sitems) ;
nms = {bout.name} ;

istr = strcmp(nms,'Protocol Name') ; 
loc = find(istr==1) ;
ProtocolName = bout(loc).data ;

istr = strcmp(nms,'Aquisition Number') ; % Note incorrect spelling!
loc = find(istr==1) ;
acqno = str2num(bout(loc).data) ;

istr = strcmp(nms,'Reconstruction Number') ; 
loc = find(istr==1) ;
recno = str2num(bout(loc).data) ;

seriesno = acqno*100 + recno ;
disp([' Computing series no. from (100.acq + rec) to be: ',num2str(seriesno)])



imarr = out.children(IMARR).children ;

if rem((length(imarr)-1),2)~=0
    warning(['Expecting even number of children in xml file, exiting.'])
    dinfo = [] ;
    return
end
nim = (length(imarr)-1)/2 ;


offset = 0 ;

% pre-allocate space
dinfo(1,nim) = struct('sl',[],'TemporalPositionIdentifier',[],...
    'RescaleSlope',[],'RescaleIntercept',[],...
    'Private_2005_100e',[], 'PixelSpacing', [], ...
    'Width',[],'Height',[],'DiffusionBValue',[],...
    'SliceOrientation',[], 'DiffusionGradientOrientation', [], ...
    'DiffusionCS', [], 'DiffGradOrientIdentifier', [], ...
    'SeriesNumber',[],'ProtocolName',[],...
    'RecOffsetBytes',[],'RecFileSize',[],'RecFileName',[]) ;

    
for iimarr = 2:2:length(imarr)
    im = iimarr / 2 ;
    
    this_item = imarr(iimarr) ;
    ch_this = this_item.children ;
    key_items = ch_this(KEY).children ;
    
    % reads the <Key> block
    %show_block(key_items) ;
    bout = read_block(key_items) ;
    nms = {bout.name} ;
    
    istr = strcmp(nms,'Slice') ;
    loc = find(istr==1) ;
    dinfo(im).sl = str2num(bout(loc).data) ;
    
    istr = strcmp(nms,'Dynamic') ;
    loc = find(istr==1) ;
    dinfo(im).TemporalPositionIdentifier = str2num(bout(loc).data) ;
    
    istr = strcmp(nms,'Grad Orient') ;
    loc = find(istr==1) ;
    dinfo(im).DiffGradOrientIdentifier = str2num(bout(loc).data) ;
    
    
    
    % reads the rest of the Image_Info
    %show_block(ch_this(3:end))
    
    bout = read_block(ch_this(3:end)) ;
    nms = {bout.name} ;
    istr = strcmp(nms,'Rescale Slope') ;
    loc = find(istr==1) ;
    dinfo(im).RescaleSlope = str2num(bout(loc).data) ;
    
    istr = strcmp(nms,'Rescale Intercept') ;
    loc = find(istr==1) ;
    dinfo(im).RescaleIntercept = str2num(bout(loc).data) ;
    
    istr = strcmp(nms,'Scale Slope') ;
    loc = find(istr==1) ;
    dinfo(im).Private_2005_100e = str2num(bout(loc).data) ;
    
    istr = strcmp(nms,'Pixel Spacing') ;
    loc = find(istr==1) ;
    dinfo(im).PixelSpacing = str2num(bout(loc).data) ;
    
    istr = strcmp(nms,'Resolution X') ;
    loc = find(istr==1) ;
    dinfo(im).Width = str2double(bout(loc).data) ;
    
    istr = strcmp(nms,'Resolution Y') ;
    loc = find(istr==1) ;
    dinfo(im).Height = str2double(bout(loc).data) ;
    
    istr = strcmp(nms,'Diffusion B Factor') ;
    loc = find(istr==1) ;
    dinfo(im).DiffusionBValue = str2num(bout(loc).data) ;
    
    istr = strcmp(nms,'Diffusion AP') ;
    loc = find(istr==1) ;
    dinfo(im).DiffusionGradientOrientation(AP) = str2num(bout(loc).data) ;
    dinfo(im).DiffusionCS = 'LPH' ;
    
    istr = strcmp(nms,'Diffusion FH') ;
    loc = find(istr==1) ;
    dinfo(im).DiffusionGradientOrientation(FH) = str2num(bout(loc).data) ;
    dinfo(im).DiffusionCS = 'LPH' ;
    
    istr = strcmp(nms,'Diffusion RL') ;
    loc = find(istr==1) ;
    dinfo(im).DiffusionGradientOrientation(RL) = str2num(bout(loc).data) ;
    dinfo(im).DiffusionCS = 'LPH' ;
    
    
    istr = strcmp(nms,'Slice Orientation') ;
    loc = find(istr==1) ;
    SliceOri = bout(loc).data;
    switch SliceOri
        case 'Transversal'
            dinfo(im).SliceOrientation = 'TRA' ;
        case 'Saggital'
            dinfo(im).SliceOrientation = 'SAG' ;
        case 'Coronal'
            dinfo(im).SliceOrientation = 'COR' ;
        otherwise
            warning(['Unknown slice orientation: ',SliceOri])
    end
    
    
    dinfo(im).SeriesNumber = seriesno ;
    dinfo(im).ProtocolName = ProtocolName ;
   
    dinfo(im).RecOffsetBytes = offset ;
    offset = offset + dinfo(im).Height*dinfo(im).Width*bpv ;
    
end

% disp(['The PAR file should be ',num2str(offset),' bytes in length.'])
for iim = 1:nim
    dinfo(iim).RecFileSize = offset ;
    dinfo(iim).RecFileName = recfn ;
end


end

function show_block(items)
disp(['  '])
 for iitem = 2:2:length(items)
     this_item = items(iitem) ;
     att = this_item.attributes ;
     att_c = {att.name} ;
     istr = strcmp(att_c,'Name') ;
     loc = istr==1 ;
     disp([att(loc).value, '  ', this_item.children.data])
 end
end

function bout = read_block(items)
nb = length([2:2:length(items)]) ;
bout(nb) = struct('name',[],'data',[]) ;

 for iitem = 2:2:length(items)
     this_item = items(iitem) ;
     att = this_item.attributes ;
     att_c = {att.name} ;
     istr = strcmp(att_c,'Name') ;
     loc = istr==1 ;
     
     bout(iitem/2).name = att(loc).value ;
     bout(iitem/2).data = this_item.children.data ;
     
 end
end
