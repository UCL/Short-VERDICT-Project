function cmHandle = tiledimCM
% tiledimCM Image Context Menu for Tiled Layout
% Apply to image handles in a TiledChartLayout created using tiledlayout
% with 'flow'. Can be used for nontiled image with reduced functionality.
%
%  cmHandle = tiledimCM
%
% Note that imcontrast does not allow an upper value above the data max so
% it may re-window when opening
%
% Example Usage:
% When tiling from consecutive DICOM frames, use columnmajor
% tl = tilelayout('flow', 'TileIndexing','columnmajor') ;
% nexttile
% h_im = imshow( ... )
% h_im.ContextMenu = tiledimCM ;
%
%
% To asign to a non-tiled layout:
%  h = findobj(gcf,'Type','image') ;
% [h.ContextMenu] = deal(tiledimCM) ;
%
% Copyright 2021, David Atkinson
%
%  See also DSELECTOR

%
% Example useful manipulations:
%  hf = src.Parent.Parent ; % Figure
%  clickedax = src.Parent.Parent.CurrentAxes ;  % The clicked axes
%   h_im = clickedax.Children   % The image in the axes
%  tl = clickedax.Parent ;  % The tiledlayout
%  axs = tl.Children ;      % The axes in the Tiled Layout
%
% Note that the axes have a Layout.Tile property. This Tile number does not
% change when axes are deleted (even thought the figure redraws)
 
% Set menu entries
cmHandle = uicontextmenu('ContextMenuOpeningFcn',@cmopening) ;
uimenu(cmHandle,'Label','Launch imcontrast','Callback',@launchImcontrast) ;
uimenu(cmHandle,'Label','Set level: [0 99th percentile]','Callback',@level099) ;
uimenu(cmHandle,'Label','Link axes','Checked','off', 'Callback',@linkax)
uimenu(cmHandle,'Label','Propagate window levels to all','Callback',@propw) ;
uimenu(cmHandle,'Label','Propagate window along row','Callback',@propwrow)
uimenu(cmHandle,'Label','Propagate window along column','Callback',@propwcol)
uimenu(cmHandle,'Label','Swap row/column major','Callback',@swapmajor)
% insert separator 
uimenu(cmHandle,'Separator','on','Label','Delete this tile (no undo)','Callback',@deleteTile) ;
uimenu(cmHandle,'Label','Delete row (no undo)','Callback',@deleteRow) ;
uimenu(cmHandle,'Label','Delete column (no undo)','Callback',@deleteCol) ;
end


% Callbacks

function cmopening(src,~)
clickedfig = src.Parent ;
htl = findobj(clickedfig,'Type','tiledlayout') ;
if isempty(htl)
    src.Children(1).Enable = 'off' ;
    src.Children(2).Enable = 'off' ;
    src.Children(3).Enable = 'off' ;
    src.Children(4).Enable = 'off' ;
    src.Children(5).Enable = 'off' ;
    src.Children(6).Enable = 'off' ;
    src.Children(7).Enable = 'off' ;
end
end

function swapmajor(src,~)
tl = src.Parent ;
switch tl.TileIndexing
    case 'columnmajor'
        tl.TileIndexing = 'rowmajor' ;
    case 'rowmajor'
        tl.TileIndexing = 'columnmajor' ;
end
end

function linkax(src,~)
hf = src.Parent.Parent ;
axs = findobj(hf,'Type','axes') ;

switch src.Checked
    case 'on'
        linkaxes(axs,'off')
        src.Checked = 'off' ;
    case 'off'
        linkaxes(axs)
        src.Checked = 'on' ;
end
end

function level099(src,~)
% LEVEL099 Set CLim from 0 to 99th percentile within the visible region
clickedax = src.Parent.Parent.CurrentAxes ;
XL = clickedax.XLim ; YL = clickedax.YLim ;

xr = ceil(XL-0.5) ;
yr = ceil(YL-0.5) ;

cdata = clickedax.Children.CData ;

clippedcdata = cdata(max(1,yr(1)):min(size(cdata,1),yr(2)),max(1,xr(1)):min(size(cdata,2),xr(2))) ;

upper = prctile(clippedcdata(:),99) ;

clickedax.CLim = [0 upper] ;
end

function propw(src,~)
% PROPW propagate window settings (CLim) to all tiles

clickedax = src.Parent.Parent.CurrentAxes ;
tl = clickedax.Parent ;


axs = tl.Children ;

cl = clickedax.CLim ;
ccl{1} = cl ;
scl = repmat(ccl,[1 length(axs)]) ;

[axs.CLim] = deal(scl{:}) ;
end

function propwrow(src,~)
% PROPWROW Propagate window level along just a row
[raxs,~] = click2rowcolax(src) ;
setrowcol(src,raxs)
end

function propwcol(src,~)
% PROPWROW Propagate window level along just a column
[~,caxs] = click2rowcolax(src) ;
setrowcol(src,caxs)
end

function setrowcol(src, axs)
clickedax = src.Parent.Parent.CurrentAxes ;
cl = clickedax.CLim ;
ccl{1} = cl ;
scl = repmat(ccl,[1 length(axs)]) ;
[axs.CLim] = deal(scl{:}) ;
end

function [raxs, caxs] = click2rowcolax(src)
% CLICK2ROWCOLAX  Return axes handles for the clicked row and column
clickedax = src.Parent.Parent.CurrentAxes ; 
tilen = clickedax.Layout.Tile ;
tl = clickedax.Parent ;

[crow, ccol] = tile2rowcol(tilen, tl) ; % clicked row and column

raxs = [] ; caxs = [] ;

% loop through tiles and store axes 
for itile = 1:length(tl.Children)
    [trow, tcol] = tile2rowcol(tl.Children(itile).Layout.Tile, tl) ;
    if trow==crow
        hdax = tl.Children(itile) ;
        raxs = [raxs hdax] ;
    end
    if tcol==ccol
        hdax = tl.Children(itile) ;
        caxs = [caxs hdax] ;
    end
end
end

function launchImcontrast(src,~)
imcontrast(src.Parent.Parent.CurrentAxes)
end

function deleteRow(src,~)
[raxs, ~] = click2rowcolax(src) ;
delete(raxs)
end

function deleteCol(src,~)
[~, caxs] = click2rowcolax(src) ;
delete(caxs)
end



function [trow, tcol] = tile2rowcol(itile, tl)
% TILE2ROWCOL Return row and column of tile number (from axes Layout Tile)
gs = tl.GridSize ; % [nrow ncol]
axs = tl.Children ;
for iax = 1 : length(axs)
    tn(iax) = axs(iax).Layout.Tile ;  % tile numbers in current figure 
end
tn_sorted = sort(tn) ; % numbers may not be sequential in axes due to deletions
sqtile = find(tn_sorted==itile) ; % sequential number
switch tl.TileIndexing
    case 'columnmajor'
        tcol = ceil(sqtile/gs(1)) ;  
        trow = sqtile - ((tcol-1)*gs(1)) ;
    case 'rowmajor'
        trow = ceil(sqtile/gs(2)) ;
        tcol = sqtile - ((trow-1)*gs(2)) ;
    otherwise
        error('Only columnmajor and rowmajor implemented')
end
end


function deleteTile(src,~)
% DELETETILE  Delete the selected tile
% This does not change the Tile number in the axes Layout property
clickedax = src.Parent.Parent.CurrentAxes ;
delete(clickedax)
end