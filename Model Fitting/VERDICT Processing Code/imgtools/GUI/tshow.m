function varargout = tshow(varargin)
%TSHOW viewing tool for tensor data
%      TSHOW, by itself, creates a new TSHOW or raises the existing
%      singleton*.
%
%
% $Id: tshow.m 207 2008-10-11 21:23:23Z ucacdat $
%      tshow(D,B0)
%      tshow(D,B0,xyz2rcs,rcs2xyz,Rg2rcs)
%      [tshow(data,data,xyz2rcs,rcs2xyz,Rg2rcs) for non diffusion ]
%        D is tensor data
%        B0 is the non diffusion weighted data (optional)
%        xyz2rcs abd rcs2xyz are [4x4] matrices that transform homogeneous
%        coordinates between the row-column-slice and DICOM coordinate
%        system. Note not adequately tested. (default to identity)
%        Rg2rcs [3 x 3] converts eigenvector in LPH to vector in rcs
%         e.g. [rcs2xyz, xyz2rcs, Rg2rcs] = gen_dicom_mat(ipp, iop, ps_rcs)
%        When computing D, use grad in LPH if using above interpretation.
%
%        keypresses call the separate function tshow_keyp:
%          A delete all non-protected surfaces
%          c camera orbit
%          R deletes all non-protected tracks
%          W deleted all non-protected ellipsoids
%          a protects all current surfaces (keep)
%          r protects all current tracks (wires)
%          w protects all current ellipsoids 
%          e draws an ellipsoid at the current point
%          t track from last point clicked in currect surface
%          p seeds multiple points around current point (area)
%          o draws ribbon on last track
%          m montage (restricts slices to those in user range)
%
%
%      H = TSHOW returns the handle to a new TSHOW or the handle to
%      the existing singleton*.
%
%      TSHOW('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TSHOW.M with the given input arguments.
%
%      TSHOW('Property','Value',...) creates a new TSHOW or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before tshow_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to tshow_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% $Id: tshow.m 207 2008-10-11 21:23:23Z ucacdat $
%
% See also: GUIDE, GUIDATA, GUIHANDLES


% Edit the above text to modify the response to help tshow

% Last Modified by GUIDE v2.5 28-Jun-2006 22:49:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @tshow_OpeningFcn, ...
                   'gui_OutputFcn',  @tshow_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before tshow is made visible.
function tshow_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to tshow (see VARARGIN)

disp('Opening figure')

cfig = hObject ;
set(cfig,'Name','tshow control') ;
UD.cfig = cfig ;
D = varargin{1} ;
locOK = isfinite(D) ;
notOK = find(locOK==0) ;
D(notOK) = 0 ;

UD.D = D ;
nz = size(UD.D,3) ;

slice = ceil(nz/2) ;
UD.slice = slice ;
UD.orbitflag = 0 ;
UD.maxorb = 0 ;

if length(varargin) > 1
    UD.B0 = varargin{2} ;
    set(handles.edit3,'String',num2str(max(max(max(UD.B0))))) ;
end

if length(varargin) > 2
    UD.xyz2rcs = varargin{3} ;
    UD.rcs2xyz = varargin{4} ;
    UD.Rg2rcs  = varargin{5} ;
else
    UD.xyz2rcs = eye(4) ;
    UD.rcs2xyz = eye(4) ;
    UD.Rg2rcs  = eye(3) ;
end

hf = figure ;
disp(['tshow opened figure ',num2str(fignum(hf))])
drawnow
hold on
axis equal
axis vis3d
grid
view(21,-50)
xlabel('x left') ; ylabel('y posterior') ; zlabel('z head') 
colormap gray
set(hf,'Tag','tshowfig');
set(hf,'UserData',UD) ;
set(hf,'KeyPressFcn','tshow_keyp') ;

set(handles.edit2,'String',num2str(fignum(hf)))
set(handles.slider1,'Max',nz)
if nz > 1
 set(handles.slider1,'Min',1,'SliderStep',[1/(nz-1) max(0.1,1/(nz-1)) ])
else
    set(handles.slider1,'Visible','off')
    set(handles.slider1,'Min',0)
   %set(handles.slider1,'Min',1,'SliderStep',[1 0.1])
end
set(handles.slider1,'Value',slice)
set(handles.edit1,'String',num2str(slice))
set(handles.popupmenu3,'Value',3)

col = [str2double(get(handles.edit4,'String')) ...
            str2double(get(handles.edit5,'String')) ...
            str2double(get(handles.edit6,'String'))] ;
set(handles.text6,'BackgroundColor',col)

set(handles.edit8,'String',num2str(size(D,2)))
set(handles.edit10,'String',num2str(size(D,1)))
set(handles.edit12,'String',num2str(size(D,3)))

% Choose default command line output for tshow
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes tshow wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = tshow_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
contents = get(hObject,'String') ;
contents{get(hObject,'Value')} ;

fig = str2double(get(handles.edit2,'String')) ;
UD = get(fig,'UserData') ;
 
fig_draw(fig, UD, handles)


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

newslice = round(get(hObject,'Value')) ;
fig = str2double(get(handles.edit2,'String')) ;
UD = get(fig,'UserData') ;

set(handles.edit1,'String',num2str(newslice)) 
UD.slice = newslice ;
set(fig,'UserData',UD) ;

fig_draw(fig,UD, handles) ;


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double

fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;

newslice = str2double( get(hObject,'String') );
newslice = max(1,newslice) ;
%newslice = min(size(UD.D,3),newslice) ;

set(handles.slider1,'Value',newslice) ;
set(hObject,'String',num2str(newslice)) ; % in case altered here
UD.slice = newslice ;
set(fig,'UserData',UD) ;


fig_draw(fig, UD, handles) 

% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%----------- 
function fig_draw(fig,UD, handles)

img = tshow_comp_image(fig, UD, handles)  ;

dplane = get(handles.popupmenu3,'Value') ;

ll(1) = str2double(get(handles.edit9,'String')) ;
ll(2) = str2double(get(handles.edit7,'String')) ;
ll(3) = str2double(get(handles.edit11,'String')) ;

ul(1) = str2double(get(handles.edit10,'String')) ;
ul(2) = str2double(get(handles.edit8,'String')) ;
ul(3) = str2double(get(handles.edit12,'String')) ;


for id = 1:3
    % indv{id} = [0.5:1:size(img,id)+1];
    indv{id} = [ll(id)-0.5:1:ul(id)+1];
    indim{id} = [ll(id):ul(id)] ;
end
indv{dplane} = UD.slice ;
indim{dplane} = 1 ; % returned as one slice 

if ndims(img) == 4
    indim{4} = [1:3] ;
end


SC = repmat(reshape(indv{2},[1 length(indv{2}) 1]),[length(indv{1}) 1 length(indv{3})]) ;
SR = repmat(reshape(indv{1},[length(indv{1}) 1 1]),[1 length(indv{2}) length(indv{3})]) ;
SS = repmat(reshape(indv{3},[1 1 length(indv{3})]),[length(indv{1}) length(indv{2}) 1]) ;

SC = squeeze(SC) ;
SR = squeeze(SR) ;
SS = squeeze(SS) ;

[nc1 nc2 nc3] = size(SC) ;
SRCS = [SR(:)' ;
        SC(:)' ;
        SS(:)' ;
        ones([1 nc1*nc2*nc3]) ] ;
  
SXYZ = UD.rcs2xyz * SRCS ;

SX = reshape(SXYZ(1,:),[nc1 nc2 nc3]) ;
SY = reshape(SXYZ(2,:),[nc1 nc2 nc3]) ;
SZ = reshape(SXYZ(3,:),[nc1 nc2 nc3]) ;


img = squeeze(img(indim{:})) ;

figure(fig)
hs = findobj(fig,'Tag','surf') ;
delete(hs) % delete non-protected surfaces
alph = get(handles.slider2,'Value') ;
hss = surf(SX,SY,SZ,img,'EdgeColor','None','Tag','surf','FaceAlpha',alph) ;




% -----------------------------------

          


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

%alph = get(hObject,'Value') ;
%hss = findobj(str2double(get(handles.edit2,'String')),'Tag','surf') ;
%set(hss,'FaceAlpha',alph) 

fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 


% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end







% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3

fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;

dplane = get(hObject,'Value') ;
nd = size(UD.D,dplane) ;
set(handles.slider1,'Max',nd)
set(handles.slider1,'Min',1,'SliderStep',[1/(nd-1) max(0.1,1/(nd-1)) ])
if isfield(UD,'seed') % seed is in YXZ
    seed = UD.seed ;
    slice = round(seed(dplane));
else
    slice = round(nd/2) ;
end

set(handles.slider1,'Value',slice)
set(handles.edit1,'String',num2str(slice))
UD.slice = slice ;
set(fig,'UserData',UD) 

fig_draw(fig, UD, handles) 


% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1

fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 





function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double

% Display max for B0
fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double
col = [str2double(get(handles.edit4,'String')) ...
            str2double(get(handles.edit5,'String')) ...
            str2double(get(handles.edit6,'String'))] ;
set(handles.text6,'BackgroundColor',col)

% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double

col = [str2double(get(handles.edit4,'String')) ...
            str2double(get(handles.edit5,'String')) ...
            str2double(get(handles.edit6,'String'))] ;
set(handles.text6,'BackgroundColor',col)

% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double

col = [str2double(get(handles.edit4,'String')) ...
            str2double(get(handles.edit5,'String')) ...
            str2double(get(handles.edit6,'String'))] ;
set(handles.text6,'BackgroundColor',col)

% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double
fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 

% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double

fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 

% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double
fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 

% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double

fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 

% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double
fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 

% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double
fig = str2double( get(handles.edit2,'String') ) ;
UD = get(fig,'UserData') ;
fig_draw(fig, UD, handles) 

% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


