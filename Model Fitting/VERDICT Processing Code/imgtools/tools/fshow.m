function hf = fshow(inp)
% FSHOW Image data display with added toolbar options.
% New version of eshow, under development
%
%  hf = fshow(inp)
%
% inp is image data or structure of UserData relating to fapp. APP
%
% David Atkinson,  D.Atkinson@ucl.ac.uk
%
% See also eshow fapp

if isstruct(inp)
    showandset(inp) ;
    return
else
    img = double(inp) ;
end

% img = imread('peppers.png') ;
% 
% img = pi* double(img)  ;

hf = figure ;
ax = gca ;

UD.Name = 'testing' ;
UD.currd3 = 1 ;
UD.currd4 = 1 ;

UD.data = img ;
UD.dispmin = min(UD.data(:)) ;
UD.dispmax = max(UD.data(:)) ;
UD.gamma = 1 ;

UD.ax = ax ;
UD.fig = hf ;

set(hf,'UserData',UD) 

showandset(UD)

end

function showandset(UD)
J = squeeze(UD.data(:,:,UD.currd3, UD.currd4)) ;
himage = imshow(J,[UD.dispmin UD.dispmax ],'Parent',UD.ax) ;
set(UD.fig,'Name',[UD.Name,' (:,:,',num2str(UD.currd3),',',num2str(UD.currd4),')']) 
tb = axtoolbar(UD.ax,{'datacursor','pan','zoomin','zoomout','restoreview'});
btn = axtoolbarbtn(tb,'push');
btn.Tooltip = 'up slice';
btn.ButtonPushedFcn = @(src, event) customcallback(src, event, 'up3');

btn = axtoolbarbtn(tb,'push');
btn.Tooltip = 'down slice';
btn.ButtonPushedFcn = @(src, event) customcallback(src, event, 'down3');

btn = axtoolbarbtn(tb,'push');
btn.Tooltip = 'up echo';
btn.ButtonPushedFcn = @(src, event) customcallback(src, event, 'up4');

btn = axtoolbarbtn(tb,'push');
btn.Tooltip = 'down echo';
btn.ButtonPushedFcn = @(src, event) customcallback(src, event, 'down4');

btn = axtoolbarbtn(tb,'push');
btn.Tooltip = 'imcontrast';
btn.ButtonPushedFcn = @(src,event) imcontrast(himage);

btn = axtoolbarbtn(tb,'state');
btn.Tooltip = 'imcontrol';
btn.ValueChangedFcn = @(src,event) imcontrol(src, event, UD);

end

function imcontrol(src, event, UD)
switch src.Value
    case 'on'
       appc = fapp(UD) ;
       UD.appc = appc ;
       UD.fig.UserData = UD ;
    case 'off'
        if isfield(UD,'appc')
          delete(UD.appc)
        end
end

end

function customcallback(src,event, dirn)
UD = get(event.Axes.Parent,'UserData') ;
switch dirn
    case 'up3'
        UD.currd3 = min(size(UD.data,3), UD.currd3 + 1);
    case 'down3'
        UD.currd3 = max(1, UD.currd3 - 1);
    case 'up4'
        UD.currd4 = min(size(UD.data,4), UD.currd4 + 1);
    case 'down4'
        UD.currd4 = max(1, UD.currd4 - 1);
end

set(event.Axes.Parent,'UserData',UD)
showandset(UD)
end


