function varargout = dmfrename(varargin)
% DMFRENAME DICOM MultiFrame file rename: adds protocol name and series no.
%
% Useful for renaming multiframe DICOMs that are output from the scanner
% as, for example, 
% I20, I21, I22
%
%
%  David Atkinson
%  University College London
%
%      DMFRENAME, by itself, creates a new DMFRENAME or raises the existing
%      singleton*.
%
%      H = DMFRENAME returns the handle to a new DMFRENAME or the handle to
%      the existing singleton*.
%
%      DMFRENAME('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DMFRENAME.M with the given input arguments.
%
%      DMFRENAME('Property','Value',...) creates a new DMFRENAME or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before dmfrename_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to dmfrename_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help dmfrename
% To DO - check SOP class and ensure in filename (for PS and XX raw)

% Last Modified by GUIDE v2.5 20-Jan-2014 21:14:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @dmfrename_OpeningFcn, ...
                   'gui_OutputFcn',  @dmfrename_OutputFcn, ...
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
end

% --- Executes just before dmfrename is made visible.
function dmfrename_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to dmfrename (see VARARGIN)

% Choose default command line output for dmfrename
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes dmfrename wait for user response (see UIRESUME)
% uiwait(handles.figure1);

set(handles.pushbutton_rename,'Enable','off')

end

% --- Outputs from this function are returned to the command line.
function varargout = dmfrename_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
end

% --- Executes on selection change in listbox_filelist.
function listbox_filelist_Callback(hObject, eventdata, handles)
% hObject    handle to listbox_filelist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox_filelist contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox_filelist
end

% --- Executes during object creation, after setting all properties.
function listbox_filelist_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox_filelist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end


function edit_folder_Callback(hObject, eventdata, handles)
% hObject    handle to edit_folder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_folder as text
%        str2double(get(hObject,'String')) returns contents of edit_folder as a double
update_namelist(handles) ;
end


% --- Executes during object creation, after setting all properties.
function edit_folder_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_folder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end


% --- Executes on button press in pushbutton_browse.
function pushbutton_browse_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if ispref('dmfrename','dir')
    defdir = getpref('dmfrename','dir');
else
    defdir = [] ;
end

folder_name = uigetdir(defdir,'Select Multi-Frame DICOM folder') ;
if folder_name == 0
    folder_string = 'Enter valid folder' ;
else
    folder_string = folder_name ;
    setpref('dmfrename','dir',folder_name)
end

set(handles.edit_folder,'String',folder_string)

update_namelist(handles) ;

end

% --- Executes on button press in pushbutton_close.
function pushbutton_close_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_close (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(handles.figure1)
end


% --- Executes on button press in pushbutton_rename.
function pushbutton_rename_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_rename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

UD = get(handles.figure1,'UserData') ;

if isfield(UD,'fnew') && isfield(UD,'fold')
    fnew = UD.fnew ;
    fold = UD.fold ;
    
    for ifile = 1:length(fnew)
       [s,mess,messid] = movefile(fold{ifile},fnew{ifile}) ; 
       if s~= 1
        errstr = [mess,'  ',messid] 
       end
    end
end

disp(['Renamed ',num2str(length(fnew)),' files.']) 
set(handles.pushbutton_rename,'Enable','off')

end

function update_namelist(handles)
set(handles.pushbutton_rename,'Enable','off')
set(handles.listbox_filelist,'String','UPDATING ...') ;
drawnow update
voldir = get(handles.edit_folder,'String') ;

imonly = get(handles.IMonly,'Value') ;
maxfs = get(handles.edit_maxfilesize,'String') ;
maxfs = str2num(maxfs) ;

if imonly == 0
    d = dir(voldir) ;
else
    % DICOM files from Philips scanner or workstation start with
    % IM_ for images, PS_ for presentation state and XX_ for raw
    d = dir(fullfile(voldir,'IM_*')) ;
end

keep = [] ;
for ident = 1:numel(d)
    if ~d(ident).isdir && d(ident).bytes > 128
        if isdicom(fullfile(voldir,d(ident).name))
            if d(ident).bytes <= maxfs*1e6
                keep = [keep ident] ;
            else
                disp(['Excluding large file: ',d(ident).name])
            end
        end
    end
end

if isempty(keep)
    set(handles.listbox_filelist,'String','No files match')
    set(handles.pushbutton_rename,'Enable','off') 
    return
end

for ifile = 1:length(keep)
   fn = d(keep(ifile)).name ;
   ffn = fullfile(voldir,fn) ;
   dinfo = dicominfo(ffn) ;
   if isfield(dinfo,'ProtocolName')
       protnm = dinfo.ProtocolName ;
       k = strfind(protnm,' ') ;
       protnm(k) = '_' ;
       k = strfind(protnm,'/') ;
       protnm(k) = '_' ;
       k = strfind(protnm,'\') ;
       protnm(k) = '_' ;
       fnprot = ['_',protnm];
   else
       protnm = '';
       fnprot = '';
   end
   if isfield(dinfo,'SeriesNumber')
       snum = dinfo.SeriesNumber ;
       snum = num2str(snum) ;
       fnsnum = ['_',snum];
   else
       snum = '';
       fnsnum = '' ;
   end
       
   addnm = [protnm,fnsnum] ;
   str = [fn,'  :  ',addnm] ;
   listb{ifile} = str ;
   newfn = [fn,fnprot,fnsnum] ;
   
   fold{ifile} = ffn ;
   fnew{ifile} = fullfile(voldir,newfn) ;
   
   set(handles.listbox_filelist,'String',{listb{:},'UPDATING ...'})
   set(handles.listbox_filelist,'Value',min(ifile+1,length(keep)))
   drawnow update

end
    
set(handles.listbox_filelist,'String',listb)

UD.fnew = fnew ;
UD.fold = fold ;

set(handles.figure1,'UserData',UD)
set(handles.pushbutton_rename,'Enable','on')


end


% --- Executes on button press in IMonly.
function IMonly_Callback(hObject, eventdata, handles)
% hObject    handle to IMonly (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of IMonly


end



function edit_maxfilesize_Callback(hObject, eventdata, handles)
% hObject    handle to edit_maxfilesize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_maxfilesize as text
%        str2double(get(hObject,'String')) returns contents of edit_maxfilesize as a double
end

% --- Executes during object creation, after setting all properties.
function edit_maxfilesize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_maxfilesize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
