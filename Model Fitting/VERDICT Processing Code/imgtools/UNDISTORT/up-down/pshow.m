function varargout = pshow(varargin)
% PSHOW MATLAB code for pshow.fig
%      PSHOW, by itself, creates a new PSHOW or raises the existing
%      singleton*.
%
%      H = PSHOW returns the handle to a new PSHOW or the handle to
%      the existing singleton*.
%
%      PSHOW('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PSHOW.M with the given input arguments.
%
%      PSHOW('Property','Value',...) creates a new PSHOW or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pshow_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pshow_OpeningFcn via varargin.
%
%      hf = PSHOW ;
%      PSHOW(MRecon, hf, ax_num)
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pshow

% Last Modified by GUIDE v2.5 08-Feb-2019 15:13:19

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pshow_OpeningFcn, ...
                   'gui_OutputFcn',  @pshow_OutputFcn, ...
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


% --- Executes just before pshow is made visible.
function pshow_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pshow (see VARARGIN)

% Choose default command line output for pshow
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes pshow wait for user response (see UIRESUME)
% uiwait(handles.figure1);


setpshow(handles, varargin{:})



% --- Outputs from this function are returned to the command line.
function varargout = pshow_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox2


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox3.
function listbox3_Callback(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox3


% --- Executes during object creation, after setting all properties.
function listbox3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox4.
function listbox4_Callback(hObject, eventdata, handles)
% hObject    handle to listbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox4


% --- Executes during object creation, after setting all properties.
function listbox4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function setpshow(handles, varargin)

M = varargin{1} ;

ax = varargin{2} ;

[fp, fn, ext] = fileparts(M.Parameter.Filename.Data) ;
set(handles.text_fn,'String',fn)

nz = size(M.Data,3) ;
if nz>1
    is3D=true ;
else
    is3D=false;
end

ikz = ceil((nz+1)/2) ;
iz=ikz ;
istack=1 ;

switch ax
    case 1
        hax = handles.axes1 ;
     
        ksp = squeeze(sum(abs(M.Data(:,:,ikz,:)),4)) ;
        image(hax,imadjust(mat2gray(ksp),[0 1], [0 1],0.4 ) ,'CDataMapping','scaled')
        daspect(hax,[1 1 1])
        colormap(hax,gray(256))
        
        lbs{1} = ['curFOV: ',num2str(M.Parameter.Scan.curFOV(istack,:))] ;
        lbs = [lbs(:)', {['KxRange: ',num2str(M.Parameter.Encoding.KxRange)]}] ;
        lbs = [lbs(:)', {['KyRange: ',num2str(M.Parameter.Encoding.KyRange)]}] ;
        if is3D, lbs = [lbs(:)', {['KzRange: ',num2str(M.Parameter.Encoding.KzRange)]}] ; end
        lbs = [lbs(:)', {['XRes: ',num2str(M.Parameter.Encoding.XRes)]}] ;
        lbs = [lbs(:)', {['YRes: ',num2str(M.Parameter.Encoding.YRes)]}] ;
        if is3D, lbs = [lbs(:)', {['ZRes: ',num2str(M.Parameter.Encoding.ZRes)]}] ; end
        lbs = [lbs(:)', {['KxOversampling: ',num2str(M.Parameter.Encoding.KxOversampling)]}] ;
        lbs = [lbs(:)', {['KyOversampling: ',num2str(M.Parameter.Encoding.KyOversampling)]}] ;
        if is3D, lbs = [lbs(:)', {['KzOversampling: ',num2str(M.Parameter.Encoding.KzOversampling)]}] ; end
        
        lbs = [lbs(:)', {['HalfScanFactors: ',num2str(M.Parameter.Scan.HalfScanFactors)]}] ;
        lbs = [lbs(:)', {['size(data): ',num2str(size(M.Data))]}] ;
        
        if is3D
            lbs = [lbs(:)', {['Only stack ',num2str(istack),' reported for curFOV and SENSE factors']}] ;
        end
        
        
        set(handles.listbox1, 'String', lbs)
        
    case 2 % pre FFT
        hax = handles.axes2 ;
        
        ksp = squeeze(sum(abs(M.Data(:,:,ikz,:)),4)) ;
        image(hax,imadjust(mat2gray(ksp),[0 1], [0 1],0.4 ) ,'CDataMapping','scaled')
        daspect(hax,[1 1 1])
        colormap(hax,gray(256))
        
        lbs{1} = ['curFOV: ',num2str(M.Parameter.Scan.curFOV(istack,:))] ;
        lbs = [lbs(:)', {['size(data): ',num2str(size(M.Data))]}] ;
        set(handles.listbox2, 'String', lbs)
        
        
    case 3 % post FFT
        hax = handles.axes3 ;
        
        img = squeeze(sum(abs(M.Data(:,:,iz,:)),4)) ;
        image(hax,imadjust(mat2gray(img),[0 1], [0 1],0.7 ) ,'CDataMapping','scaled')
        daspect(hax,[1 1 1])
        colormap(hax,gray(256))
        
        lbs{1} = ['curFOV: ',num2str(M.Parameter.Scan.curFOV(istack,:))] ;
        lbs = [lbs(:)', {['XRange: ',num2str(M.Parameter.Encoding.XRange)]}] ;
        lbs = [lbs(:)', {['YRange: ',num2str(M.Parameter.Encoding.YRange)]}] ;
        if is3D, lbs = [lbs(:)', {['ZRange: ',num2str(M.Parameter.Encoding.ZRange)]}] ; end
        lbs = [lbs(:)', {['SENSEFactor: ',num2str(M.Parameter.Scan.SENSEFactor(istack,:))]}] ; 
        lbs = [lbs(:)', {['size(data): ',num2str(size(M.Data))]}] ;
        lbs = [lbs(:)', {['cur pix: ',num2str(M.Parameter.Scan.curFOV(istack,1)/size(M.Data,1)), ...
            '  ', num2str(M.Parameter.Scan.curFOV(istack,2)/size(M.Data,2)) ]}] ;
        
        set(handles.listbox3, 'String', lbs)
        
   case 4 % post FFT
        hax = handles.axes4 ;
        
        img = squeeze(sum(abs(M.Data(:,:,iz,:)),4)) ;
        image(hax,imadjust(mat2gray(img),[0 1], [0 1],0.7 ) ,'CDataMapping','scaled')
        daspect(hax,[1 1 1])
        colormap(hax,gray(256))
        
        lbs{1} = ['curFOV: ',num2str(M.Parameter.Scan.curFOV(istack,:))] ;
        lbs = [lbs(:)', {['size(data): ',num2str(size(M.Data))]}] ;
        lbs = [lbs(:)', {['cur pix: ',num2str(M.Parameter.Scan.curFOV(istack,1)/size(M.Data,1)), ...
            '  ', num2str(M.Parameter.Scan.curFOV(istack,2)/size(M.Data,2)) ]}] ;
        
        set(handles.listbox4, 'String', lbs)
        
    case 5 % post FFT
        hax = handles.axes5 ;
        
        img = squeeze(sum(abs(M.Data(:,:,iz,:)),4)) ;
        image(hax,imadjust(mat2gray(img),[0 1], [0 1],0.7 ) ,'CDataMapping','scaled')
        daspect(hax,[1 1 1])
        colormap(hax,gray(256))
        
        lbs{1} = ['curFOV: ',num2str(M.Parameter.Scan.curFOV(istack,:))] ;
        
        lbs = [lbs(:)', {['size(data): ',num2str(size(M.Data))]}] ;
        lbs = [lbs(:)', {['cur pix: ',num2str(M.Parameter.Scan.curFOV(istack,1)/size(M.Data,1)), ...
            '  ', num2str(M.Parameter.Scan.curFOV(istack,2)/size(M.Data,2)) ]}] ;
        
        
        set(handles.listbox5, 'String', lbs)
        
    otherwise
        warning(['Ax not implemented.'])
end




return


% --- Executes on selection change in listbox5.
function listbox5_Callback(hObject, eventdata, handles)
% hObject    handle to listbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox5 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox5


% --- Executes during object creation, after setting all properties.
function listbox5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
