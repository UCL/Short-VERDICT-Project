function sviewer(v,m_in, opts)
% SVIEWER
% 
% sviewer(v,m)
% sviewer(v, [])  no geometry information
% sviewer(v,m, Name,value, ...)
%
% Slices in v should be parallel.
%  v can be 3D or 4D. The 3rd dimension should represent space.
%  m must either have a field geom, or be a geom structure itself. If m is
%  empty, geometry information is not displayed and a fake
%  FrameOfReferenceUID is used.
%
% Use up/down arrows or scrollwheel for slices, left/right for 4th dimension
%
% Name, Value pairs:
%   CLim         [1 2] applied to all dimensions, or, [nd4 2], or,
%                cell array with values 'T2W', 'b-low', 'b-high' either 
%                size [1 2] or [nd4 2]
%   isrgb        {false} | true
%   indexD4      {[]}   4th dimension index to display (defaults to central)
%   d4annotation {{}} Cell array of string vectors, of length dimension 4
%     for annotation in viewer (default is index number)
%
% Example
% dinfo = getdinfo(dselector) ; 
% [v,m] = d2mat(dinfo,{'slice','effTE'},'op','fp') ;
% sviewer(v,m,Name='multi-echo', CLim=[0 1000])
%
% David Atkinson
%
% See also findclosestslice setlink

arguments
    v  double {mustBeNumericOrLogical}  % needs to be double for colormap editor
    m_in
    opts.Name {mustBeTextScalar} = '' % later defaults to first variable name
    opts.CLim 
    opts.inputroi
    opts.isrgb = false
    opts.indexD4 = [] ;
    opts.d4annotation = {} ;
    opts.probe = [] ;
end

if opts.isrgb
    % Fix v to be 5D with last dim size 3
    if size(v,ndims(v)) ~= 3
        warning('MATLAB:sviewer:rgbInput','For RGB, input last dim must be size 3')
    end

    if ndims(v) == 3
        v = reshape(v,[size(v,1) size(v,2) 1 1 3]) ;
    elseif ndims(v) == 4
        v = reshape(v,[size(v,1) size(v,2) size(v,3) 1 3]) ;
    elseif ndims(v) > 5
        warning('MATLAB:sviewer:inputSize','Input matrix has too many dimensions')
    end

    v = mat2gray(v) ;

else
    if ndims(v) > 4
        warning('Input has more than 4 dimensions, attempting to squeeze')
        v = squeeze(v) ;
        if ndims(v) > 4
            warning('MATLAB:sviewer:inputSize','Input matrix has too many dimensions')
        end
    end
end

if isempty(opts.probe)
    opts.probe = v ;
else
    % needs tidying up
    % Could let probe apply to all D4, but would need to warn
    ndp = ndims(opts.probe) ;
    if ~isequal(size(v, [1:ndp]), size(opts.probe))
        warning(['probe and v have different sizes'])
    end
end

if isempty(m_in)
    warning(['m_in is empty - creating a fake axial geometry'] )
    m = fake_geom(v) ;
    donotdisplayori = true ;
else
  if ~isfield(m_in,'geom'), m.geom = m_in ; else, m = m_in ; end  % m was geom on input
  donotdisplayori = false ;
end

if isempty(opts.Name)
    opts.Name = inputname(1);
end

if ~isempty(opts.d4annotation) && length(opts.d4annotation) ~= size(v,4) 
    opts.d4annotation = {} ;
end

hf = figure('Name',opts.Name,'WindowScrollWheelFcn',@figScroll, ...
    'WindowKeyPressFcn',@KeyPress, ...
    'Tag','sviewer_fig') ;
disp(['Opened Figure ',num2str(hf.Number),': ',hf.Name])

% Context Menu handle that will be assigned to every image.
cmHandle = uicontextmenu(hf) ;
cmHandle.ContextMenuOpeningFcn = @cmopening ;
if ~opts.isrgb
    uimenu(cmHandle,'Label','imcontrast','Callback',@window) ;
    uimenu(cmHandle,'Label','set CLim','Callback',@setclim) ;
else
    uimenu(cmHandle,'Label','imcontrast','Callback',@window,'Enable','off') ;
    uimenu(cmHandle,'Label','set CLim','Callback',@setclim,'Enable','off') ;
end

uimenu(cmHandle,'Label','Link all spatial axes','Callback',{@setlink, 'all'});
uimenu(cmHandle,'Label','Select linked axes (in-plane)','Callback',{@setlink, 'axes'});
uimenu(cmHandle,'Label','Unlink axes (in-plane)','Callback',{@setlink, 'axes unlink'});
uimenu(cmHandle,'Label','Select figs to link CLim of axes','Callback',{@setlink, 'CLim'});
uimenu(cmHandle,'Label','unlink CLim','Callback',{@setlink, 'CLim unlink'});
uimenu(cmHandle,'Label','Select linked slices (Zlink)','Callback',{@setlink, 'zlink'});
uimenu(cmHandle,'Label','Draw polygon ROI','Callback',{@drawroi_poly});
uimenu(cmHandle,'Label','Draw line ROI','Callback',{@drawroi_line});
uimenu(cmHandle,'Label','Draw Crosshair','Callback',{@drawroi_crosshair})
uimenu(cmHandle,'Label','ROI stats','Callback',{@roistats}) ;
uimenu(cmHandle,'Label','Add linked ROI','Callback',{@addlinkedroi});
uimenu(cmHandle,'Label','Add linked Crosshair','Callback',{@addlinkedCrosshair});
uimenu(cmHandle,'Label','Import RTstruct','Callback',{@importrtstruct});
uimenu(cmHandle,'Label','Toggle data probe','Callback',{@toggleDataProbe});
uimenu(cmHandle,'Label','Montage selected slices','Callback',{@montageSlices});
uimenu(cmHandle,'Label','Sort figures','Callback',{@sortFigures});

UD.cmHandle = cmHandle ;

ha = axes(hf) ;

% "slice" refers to 3rd dimension and "d4" to 4th
% Implicit assumption that the 3rd dimension is spatial.
nslice = size(v,3) ;
nd4    = size(v,4) ;

dslice = ceil(nslice/2) ;
if ~isempty(opts.indexD4) && opts.indexD4 > 0 && opts.indexD4 <=nd4
    dd4 = opts.indexD4 ;
else
    dd4 = ceil(nd4/2) ;
end

oinfo = ori_info(m.geom(1).IOP,'geom',m.geom) ;
if donotdisplayori
    oinfo.positive_z_str = '';
    oinfo.west_str = '' ;
    oinfo.east_str = '' ;
end

anslice = annotation(hf,'textbox',[0.01 0.01 0.1 0.1],...
            'String',[oinfo.positive_z_str,' ',num2str(dslice), ...
            '/',num2str(nslice)], ...
            'Color',[1 0 0],'FontSize',14) ;

and4 = annotation(hf,'textbox',[0.01 0.1 0.1 0.1],...
            'String',[num2str(dd4),'/',num2str(nd4)], ...
            'Color',[0 0 1],'FontSize',14) ;

anwest = annotation(hf, 'textbox',[0.01 0.5 0.1 0.1], ...
    'String',oinfo.west_str,'Color',[0 0 0],'FontSize',14) ;

aneast = annotation(hf, 'textbox',[0.9 0.5 0.1 0.1], ...
    'String',oinfo.east_str,'Color',[0 0 0],'FontSize',14) ;


UD.dd4 = dd4;
UD.nd4 = nd4;
UD.d4annotation = opts.d4annotation ;

UD.dslice = dslice ;
UD.nslice = nslice ;
UD.anslice = anslice ;
UD.and4 = and4 ;
UD.anposz = oinfo.positive_z_str ;
UD.m = m ;
UD.ha = ha ;
UD.iszlinked = false ;
UD.probe = opts.probe ;

XData = m.geom(1).XData;
YData = m.geom(1).YData;
% warm-up axes
% using ones to avoid NaN issues in range
himshow = imshow(ones(size(v(:,:,1,1,1))),[],'XData',XData,'YData',YData,'Parent',ha) ;


set(himshow,'Visible','off')

%apply all slices to axes, but make invisible
ip = {'XData',XData,'YData',YData,'CDataMapping','scaled','Visible','off','CData'} ;

for islice = 1:size(v,3)
    for id4 = 1:nd4
        hind = sub2ind([nslice nd4],islice, id4) ;
        if opts.isrgb
            him(hind) = image(ha,ip{:},squeeze(v(:,:,islice,id4,:))) ;
        else
            him(hind) = image(ha,ip{:},v(:,:,islice,id4)) ;
        end
    end
end

UD.him = him ;
UD.slice2ROIHandle = cell([1 nslice]);

if isfield(opts,'inputroi') && ~isempty(opts.inputroi)
    for islice = 1: nslice
        rois = opts.inputroi{islice} ;
        for iroi = 1:length(rois)

            roi_this = images.roi.Polygon(ha, Position=rois(iroi).POS, ...
                Color=rois(iroi).Color, MarkerSize=1) ;
            
            UD.slice2ROIHandle{1, islice} = cat(2, UD.slice2ROIHandle{1, islice}, roi_this) ;
        end
    end
end


% input CLim
%  setting CLim for each nd4 for each figure.
% [1 2] - apply to all d4
% [nd4 2] - directly use
% Character inputs
%  '' use stretchlim [0 98]
%  'T2W' stretchlim [0 90]
%  'b-low' strechlim [0 90]
%  'b-high' strechlim [10 100]
% {} length 1 char array - apply to all d4 of figure
% {} length d4 - apply to each d4

clim_perd4 = repmat([0 1], [nd4 1]) ;
if ~opts.isrgb
    if ~isfield(opts,'CLim')
        clim_perd4 = repmat([min(v(:)) prctile(v(:),98) ], [nd4 1]) ;
    else
        if iscell(opts.CLim)
            cellCLim = opts.CLim;
            if length(cellCLim) == 1
                cellCLim = repmat(cellCLim,[nd4 1]) ;
            end
            for id4=1:nd4
                temp = v(:,:,:,id4) ;
                switch cellCLim{id4}
                    case 'T2W'
                        clim_perd4(id4,:) = [0 prctile(temp(:),95) ] ;
                    case 'b-low'
                        clim_perd4(id4,:) = [0 prctile(temp(:), 95)] ;
                    case 'b-high'
                        clim_perd4(id4,:) = [prctile(temp(:),10) max(temp(:))] ;
                    otherwise
                        error('Input CLim not recognised')
                end
            end
        else
            if size(opts.CLim,2) ~= 2
                error('Input CLim should have 2 entries')
            end
            if size(opts.CLim,1) == 1
                clim_perd4 = repmat(opts.CLim,[nd4 1]) ;
            elseif size(opts.CLim,1) == nd4
                clim_perd4 = opts.CLim ;
            else
                error('Input CLim must have 1 or nd4 entries') ;
            end
        end
    end
end % isrgb
UD.clim_perd4 = clim_perd4 ;

hf.UserData = UD ;

show_slice(hf)


% % Delete
% if isfield(opts,'CLim')
%     fakesrc.Parent.Parent = hf ; % Fakes context menu selection for CLim
%     setclim(fakesrc,[],opts.CLim)
% else
%     clim = [min(v(:)) prctile(max(v(:)),98) ] ;
%     if ~opts.isrgb
%         fakesrc.Parent.Parent = hf ; % Fakes context menu selection for CLim
%         setclim(fakesrc,[],clim)
%     end
% end

end


function show_slice(hf)
% make frame dslice. dd4 visible, update annotation
UD = get(hf,'UserData');
hand = UD.him ;
if any(~isgraphics(hand,'image'))
    hf.Name = 'Invalid image' ;
    return
end
soff{1} = 'off';  % make all images not visible first
cstr = repmat(soff,[1 length(UD.him)]) ;
[UD.him.Visible] = deal(cstr{:})  ;
hind = sub2ind([UD.nslice UD.nd4],UD.dslice, UD.dd4) ;
UD.him(hind).Visible = 'on' ;
        
UD.him(hind).ContextMenu = UD.cmHandle ;
UD.ha.CLim = UD.clim_perd4(UD.dd4,:) ;

% annotation boxes
UD.anslice.String = [UD.anposz,' ',num2str(UD.dslice),'/',num2str(UD.nslice)] ;
if isempty(UD.d4annotation) || length(UD.d4annotation) ~= UD.nd4
    UD.and4.String = [num2str(UD.dd4),'/',num2str(UD.nd4)] ;
else
    UD.and4.String = UD.d4annotation{UD.dd4} ;
end

% Turn off any ROIs
hROIs = [UD.slice2ROIHandle{:}]  ;
if ~isempty(hROIs) && all(isgraphics(hROIs))
    rstr = repmat(soff,[1 length(hROIs)]) ;
    [hROIs.Visible] = deal(rstr{:}) ;
end

% Turn on ROIs in displayed slice
hsliceROI = UD.slice2ROIHandle{UD.dslice} ;
if ~isempty(hsliceROI) && all(isgraphics(hsliceROI))
    ron{1} = 'on' ;
    ronstr = repmat(ron,[1 length(hsliceROI)]) ;
    [hsliceROI.Visible] = deal(ronstr{:}) ;
end

end
        

function figScroll(src, event)
% scroll wheel to udate slice
% See also up/down arrow in KeyPress
UD = get(src,'UserData') ;
if event.VerticalScrollCount > 0
    if UD.dslice < UD.nslice
        UD.dslice = UD.dslice + 1;
    end
elseif event.VerticalScrollCount < 0
    if UD.dslice > 1
        UD.dslice = UD.dslice - 1;
    end
end
UD.clim_perd4(UD.dd4,:) = UD.ha.CLim ;
set(src,'UserData',UD)
if UD.iszlinked
    cent3D = centrein3D(src) ;
else
    show_slice(src)
end
end

function KeyPress(src, KeyData)
hf = src ;
UD = get(hf,'UserData') ;
UD.clim_perd4(UD.dd4,:) = UD.ha.CLim ;
switch KeyData.Key
    case 'uparrow'
        if UD.dslice < UD.nslice
            UD.dslice = UD.dslice + 1;
        end
    case 'downarrow'
        if UD.dslice > 1
            UD.dslice = UD.dslice - 1;
        end
    case 'leftarrow'
        if UD.dd4 > 1
            UD.dd4 = UD.dd4 - 1;   
        end
    case 'rightarrow'
        if UD.dd4 < UD.nd4
            UD.dd4 = UD.dd4 + 1;
        end
    otherwise
end

set(hf,'UserData',UD)

if UD.iszlinked
    cent3D = centrein3D(hf) ;
else
    show_slice(hf)
end

end

function cent3D = centrein3D(hf)
% Finds 3D centre and updates all zlinked figures

ha = findobj(hf,'Type','axes');
xc = (ha.XLim(2) - ha.XLim(1))/2 ; % centre of current displayed image
yc = (ha.YLim(2) - ha.YLim(1))/2 ;

UD = get(hf,'UserData') ;
geom = UD.m.geom ;

ipp = geom(UD.dslice).IPP ;
iop = geom(UD.dslice).IOP ;
XData = geom(UD.dslice).XData ;
YData = geom(UD.dslice).YData ;

% convert centre point to 3D LPH coordinate
cent3D = ipp + (xc-XData(1))*iop(1:3) + (yc-YData(1))*iop(4:6) ;

% get all figures that are slice linked,
% loop through and re-display

UDg = get(groot,'UserData') ;
zlhfigs = UDg.zlhfigs ; % zlinked figure handles

for izl = 1:length(zlhfigs)
    if isgraphics(zlhfigs(izl)) % in case user deleted a figure after linking
        UDf = get(zlhfigs(izl),'UserData') ;
        ind_slice = findclosestslice(cent3D, UDf.m.geom) ;
        UDf.dslice = ind_slice ;
        set(zlhfigs(izl),'UserData',UDf)
        show_slice(zlhfigs(izl))
    end
end
end


function window(src, event)
% ha = findobj(src.Parent.Parent,'Type','axes');
hf = src.Parent.Parent ;
UD = hf.UserData ;
hind = sub2ind([UD.nslice UD.nd4],UD.dslice, UD.dd4) ;
htool = imcontrast(UD.him(hind));
hadjb = findall(htool,'Tag','adjust data button') ;
if ishandle(hadjb) % in case user quits on initial warning about data range
    hadjb.Visible = 'off' ;
end
end

function setclim(src, ~, climin)
hf  = src.Parent.Parent ;
UD = hf.UserData ;

ha = UD.ha ;
clim_this = ha.CLim ;

if nargin < 3
    newclim = input(['Current CLim is [',num2str(clim_this(1)),' ',num2str(clim_this(2)),']. Specify new: ']) ;
else
    newclim = climin ;
end

if length(newclim) ~= 2 || newclim(1)>=newclim(2)
    disp('CLim must be 2-element vector with first entry less than second.')
    return
end

ha.CLim = newclim ;

end

function drawroi_poly(src,event)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;
roi = drawpolygon ;
roi.Tag = 'sviewer_roi' ;
end

function drawroi_line(src,event)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;
roi = drawline ;
roi.Tag = 'sviewer_roi' ;
end

function drawroi_crosshair(src,event)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;
roi = drawcrosshair('EdgeAlpha',0.5) ;
roi.Tag = 'sviewer_roi' ;
end


function roistats(src, event)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;
rois = findobj(ha,'Tag','sviewer_roi') ;

hind = sub2ind([UD.nslice UD.nd4],UD.dslice, UD.dd4) ;
if isfield(UD,'probe')
    cdata = UD.probe(:, :, UD.dslice, UD.dd4) ;
else
    cdata = UD.him(hind).CData ;
end

pixArea = UD.m.geom(UD.dslice).PixelSpacing_HW(1) * ...
    UD.m.geom(UD.dslice).PixelSpacing_HW(2) ;


for iroi = 1:length(rois)
    roiType = rois(iroi).Type ;
    switch roiType
        case 'images.roi.line'
            posline = rois(iroi).Position ;
            dx = abs(posline(2,1) - posline(1,1))  ;
            dy = abs(posline(2,2) - posline(1,2))  ;

            disp(['Line length: ',num2str(norm([dx dy]))])
        case 'images.roi.polygon'
            bw = createMask(rois(iroi), UD.him(hind)) ;
            dat = cdata(bw) ;
            disp(['npix: ',num2str(sum(bw(:))),...
                ', area: ',num2str(bwarea(bw)*pixArea), ...
                ', sl: ',  num2str(UD.dslice), ...
                ', mean: ',num2str(mean(dat(:))),...
                ', median: ',num2str(median(dat(:))), ...
                ', min: ',num2str(min(dat(:))), ...
                ', max: ',num2str(max(dat(:))),...
                ])
        case 'images.roi.Crosshair'
            pos = rois(iroi).Position ;
            disp(['Crosshair position: ',num2str(pos(1)),' ',num2str(pos(2))])
        otherwise
            error(['Unknown ROI type: ',roiType])
    end
end % iroi

end

function addlinkedroi(src,event)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;
otherrois = findall(groot,'Tag','sviewer_roi') ;
if isempty(otherrois)
    disp('No rois found')
    return
end
if length(otherrois) > 1
    disp(['More than ROI found, will use first'])
end
origroi = otherrois(1) ;
roi = images.roi.Polygon(ha)  ;

roi.Position = origroi.Position ;
UD.hlroi = linkprop([roi origroi],'Position') ;
hf.UserData = UD ;
end

function addlinkedCrosshair(src,event)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;
otherrois = findall(groot,'Tag','sviewer_roi') ;
if isempty(otherrois)
    disp('No rois found')
    return
end
for iroi = 1: length(otherrois)
    if otherrois(iroi).Type == 'images.roi.crosshair' 
        hc(iroi) = drawcrosshair(ha,'Position',otherrois(iroi).Position,'EdgeAlpha',0.5) ;
    end
end

UD.hlcrosshair = linkprop([otherrois(iroi) hc],'Position') ;
hf.UserData = UD ;

end



function importrtstruct(src,event)
rtfile = getdfiles(dselector,'Select RTStruct file') ;
rtinfo = dicominfo(rtfile{1}) ;
rtContours = dicomContours(rtinfo) ;

uniqueRTFrameOfReferenceUID = frameOfRefCheck(rtinfo, rtContours) ;

hf=src.Parent.Parent ;
ax = findobj(hf,'Type','Axes') ; % replace with UD.ha?
UD = hf.UserData ;

if ~isfield(UD.m.geom,'FrameOfReferenceUID') 
    warning('MATLAB:sviewer:NoFOfRefUID','No FrameOfRefUID in image data')
    FrameOfRefUID = '' ;
else
    FrameOfRefUID = UD.m.geom(1).FrameOfReferenceUID ;
end

if ~strcmp(uniqueRTFrameOfReferenceUID,FrameOfRefUID)
    warning('MATLAB:sviewer:InconsistentFrameOfRefUID', ...
        'FrameOfReferenceUIDs for contours and images do not match. Possible geometrical error in display')
end


[rt2Slice, slice2RT, rt2RefSOPInstanceUID] = indexRT(rtContours, UD.m.geom, rtinfo) ;
UD.slice2RT = slice2RT ;

nSlices = UD.nslice ;

% On each slice (currently irrespective of 4th dimension), create a 2D ROI
% if needed and store the handle in cell array UD.slice2ROIHandle 

slice2ROIHandle = cell([1 nSlices]) ;

for iSlice = 1:nSlices
    
    rtinslice = slice2RT{iSlice} ; % Get ROI and Contour numbers in slice

    for iRT = 1:size(rtinslice,1)
        iROI     = rtinslice(iRT,1) ;
        iContour = rtinslice(iRT,2) ;

        % currently not doing anything with this (might be used to check
        % ROI is on correct DICOM)
        refuiddat = rt2RefSOPInstanceUID{iROI};
        refUID = refuiddat{iContour};

        contdat = rtContours.ROIs.ContourData{iROI} ;
        inputCoord3D = contdat{iContour} ;

        [roiCoord2D, distances] = coord2D3D(inputCoord3D,UD.m.geom(iSlice),'type2D','2Dmm') ;

        % Create MATLAB 2D ROI
        ROI = images.roi.Polygon(ax,'Position',roiCoord2D) ;
        ROI.Color = rtContours.ROIs.Color{iROI}/255 ;
        ROI.Label = rtContours.ROIs.Name{iROI} ;
        ROI.MarkerSize = 4 ;
        ROI.Tag = 'sviewer_roiFromRTstruct' ;

        slice2ROIHandle{1,iSlice} = cat(2,slice2ROIHandle{1,iSlice}, ROI) ;
    end

end

UD.slice2ROIHandle = slice2ROIHandle ;
hf.UserData = UD ;

show_slice(hf)

end


function cmopening(src,~)
% clickedfig = src.Parent ;
% htl = findobj(groot,'Tag','sviewer_fig') ;
% hother = setdiff(htl, clickedfig) ;
% for iother = 1:length(hother)
%     uimenu(src,"Text",['Fig ',hother(iother).Name,' ',num2str(hother(iother).Number)])
% end

end

function toggleDataProbe(src,~)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;

wbmfn = get(hf,'WindowButtonMotionFcn') ;
if isempty(wbmfn)
    set(hf,'WindowButtonMotionFcn',@motionDataProbe)
    annprobe = annotation(hf,'textbox',[0.01 0.9 0.1 0.1],...
               'String',['val'], ...
               'Color',[0 0 1],'FontSize',14) ;
    UD.annprobe = annprobe ;
else
    set(hf,'WindowButtonMotionFcn','')
    delete(UD.annprobe)
end
hf.UserData = UD;

end

% -   - - - -
function sortFigures(src,~)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;

pos = hf.Position ;
fw = pos(3) ;
fh = pos(4) ;
fx = pos(1) ;
fy = pos(2) ;

allfig = findall(groot,'Tag','sviewer_fig') ;

for ifig = 1:length(allfig)
    if allfig(ifig) ~= hf
        fx = fx + fw ;
        allfig(ifig).Position = [fx fy fw fh] ;
    end
end

end
% -   - - - -

function montageSlices(src,~)
hf=src.Parent.Parent ;
UD = hf.UserData ;
ha = UD.ha ;
clim = ha.CLim ;

slicesToMontageInput = input(['Enter slices to montage, e.g. 1:', ...
    num2str(UD.nslice),' or {1:10, 3}']) ;
if iscell(slicesToMontageInput)
    if length(slicesToMontageInput) ~=2
        warning(['For cell array input, must be slices, then dim 4'])
    end
    slicesToMontage = slicesToMontageInput{1} ;
    altd4 = slicesToMontageInput{2} ;
    alternate = true ;
    afac = 2;
    altClim = clim ; % !! Was hoping for another clim

else
    slicesToMontage = slicesToMontageInput ;
    alternate = false ;
    afac=1;
end

if min(slicesToMontage) < 1 || max(slicesToMontage) > UD.nslice
    warning('Slice range beyond data')
end

hftile = figure('Name','montage') ;
t = tiledlayout(hftile, 'flow','Padding','tight','TileSpacing','none') ;
axlink = [] ;
for itile = 1:afac*length(slicesToMontage)
    hatile = nexttile ;
    axlink = [axlink hatile] ;
    if alternate
        islice = ceil(itile/2) ;
    else
        islice = itile ;
    end
    if alternate && rem(itile,2)==0
        hind = sub2ind([UD.nslice UD.nd4],slicesToMontage(islice), altd4) ;
    else
        hind = sub2ind([UD.nslice UD.nd4],slicesToMontage(islice), UD.dd4) ;
    end
    himThis = UD.him(hind) ;

    imshow(himThis.CData,'XData',himThis.XData,'YData',himThis.YData,'Parent',hatile)
    if alternate && rem(itile,2)==0
        hatile.CLim = altClim ;
    else
        hatile.CLim = clim ;
    end
    hatile.XLim = ha.XLim ;
    hatile.YLim = ha.YLim ;
end
linkaxes(axlink)

end
% -   - - - -


function motionDataProbe(src, event)
hf = src ;
UD = hf.UserData ;
ha = UD.ha ;

% Axes have XLim and YLim, not XData, YData. Get these from the image child
hImChild = findobj(ha, 'Type', 'image') ;
if isempty(hImChild)
    return
end

XData = get(hImChild(1),'XData') ;
YData = get(hImChild(1),'YData') ;

cursorPos = get(ha, 'CurrentPoint') ;

% Convert to image coordinates
col = ceil(0.5 + (cursorPos(1,1) - XData(1)) /...
    UD.m.geom(UD.dslice).PixelSpacing_HW(2) ) ;

row = ceil(0.5 + (cursorPos(1,2) - YData(1)) /...
    UD.m.geom(UD.dslice).PixelSpacing_HW(1) ) ;

if row > 0 && row <= UD.m.geom(UD.dslice).Height && ...
        col > 0 && col <= UD.m.geom(UD.dslice).Width
    astr{2} = ['(',num2str(row),', ',num2str(col),')'] ;
    val = UD.probe(row, col, UD.dslice, UD.dd4) ;
    astr{1} = num2str(val) ;

    UD.annprobe.String = astr ;

end


end

function m = fake_geom(v) 
% FAKE_GEOM Fakes a geom for the input volume
[ny, nx, nz] = size(v,[1 2 3]) ;

foruid = dicomuid ;

for iz = 1:nz
    geom(iz).IOP = [1 0 0 0 1 0];
    geom(iz).IPP = [0 0 iz] ;
    geom(iz).Height = ny ;
    geom(iz).Width = nx ;
    geom(iz).PixelSpacing_HW = [ 1 1];
    geom(iz).SliceThickness = 1 ;
    geom(iz).XData = [1 nx] ;
    geom(iz).YData = [1 ny] ;
    geom(iz).MRAqType = 'fake' ;
    geom(iz).FrameOfReferenceUID = foruid ;
    geom(iz).source = 'fake geom' ;
end

m.geom = geom ;

end

% s=sliceViewer(v)
% 
% 
% midslice = ceil(nslice/2) ;
% 
% figure
% hold on
% for islice = 1:nslice
%     [L,P,H] = dicom2intrinsic(m.geom(islice),'output', 'LPHcoords') ; 
% 
%     h(islice) = surf(L,P,H,v(:,:,islice),'EdgeColor','None', ...
%     'HandleVisibility','off') ;
% end
% colormap gray
% xlabel('L'),ylabel('P'),zlabel('H')

% For true axial view:
% ha = gca 
% view(ha,[0 0 1])
% ha.CameraUpVector = [0 -1 0]; 
