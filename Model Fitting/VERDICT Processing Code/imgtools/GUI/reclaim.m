function varargout = reclaim(varargin)
% RECLAIM MATLAB code for reclaim.fig
%      RECLAIM, by itself, creates a new RECLAIM or raises the existing
%      singleton*.
%
%      H = RECLAIM returns the handle to a new RECLAIM or the handle to
%      the existing singleton*.
%
%      RECLAIM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RECLAIM.M with the given input arguments.
%
%      RECLAIM('Property','Value',...) creates a new RECLAIM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before reclaim_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to reclaim_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help reclaim

% Last Modified by GUIDE v2.5 25-Nov-2018 22:12:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @reclaim_OpeningFcn, ...
                   'gui_OutputFcn',  @reclaim_OutputFcn, ...
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


% --- Executes just before reclaim is made visible.
function reclaim_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to reclaim (see VARARGIN)

% Choose default command line output for reclaim
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes reclaim wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = reclaim_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

    


function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


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


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Browse for XML/REC
folder_xml = pref_uigetdir('xmlanon', 'folder') ;
set(handles.edit1,'String',folder_xml)
update_name_in_file(handles) ;






% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


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


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Call xmlanon

npn = char(get(handles.edit3,'String')) ;
if iscell(npn)
    xmlanon(npn{1})
else
    xmlanon(npn)
    
end



function update_name_in_file(handles)
% See also XMLANON
folder = char(get(handles.edit1,'String')) ;

dxml = dir(fullfile(folder,'*.xml')) ;
nxml = length(dxml) ;

dpar = dir(fullfile(folder,'*.PAR')) ;
npar = length(dpar) ;

set(handles.text7,'String',[num2str(nxml),' xml and ',num2str(npar),' PAR files'])


for ixml = 1:nxml
    [fid, msg] = fopen(fullfile(folder,dxml(ixml).name),'rt') ;
    if fid==-1
        warning(msg) ;
    end
    
    npnline = 0 ;
    tline = fgets(fid);
    while ischar(tline)
      match_PN = strfind(tline, 'Patient Name');
      if ~isempty(match_PN)
          % Assumes name is between the two strings below
          sst = 14 + strfind(tline,'Type="String"');
          sfn = strfind(tline,'</Attribute>') - 1;
        
          if isempty(sst) || isempty(sfn)
              warning(['Problem with finding Patient Name string'])
              overwrite = false ;
          end
          
          pn{ixml} = tline(sst:sfn) ;
          
          npnline = npnline+1 ;
      end
      
      tline = fgets(fid);
    end
    
    if npnline ~= 1
        warning(['Should be exactly 1 Patient Name line in: ',dxml(ixml).name])
    end
    
    fclose(fid) ;
end
if length(unique(pn)) ~= 1
    set(handles.text6,'String','!! No single unique patient name !!')
else
    set(handles.text6,'String', pn{1})
end


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

pn = get(handles.text6,'String')
set(handles.edit3,'String',pn)


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

xmlanon


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Make parent and sub folders within containg folder

stdname = char(get(handles.edit3,'String')) ; % char in case cell returned
contain_folder = char(get(handles.edit4, 'String')) ;
parent_folder = fullfile(contain_folder, stdname) ; 
sub_folder_xml = fullfile(parent_folder,[stdname,'_XML']) ;
sub_folder_dicom = fullfile(parent_folder,[stdname, '_DICOM']) ;

local_mkdir(parent_folder) ;
local_mkdir(sub_folder_xml) ;
set(handles.text10,'String',sub_folder_xml)
set(handles.text11,'String',sub_folder_dicom)
set(handles.text13,'String',parent_folder)

local_mkdir(sub_folder_dicom) ;

update_listbox(handles)



    
function local_mkdir(dirn)
[status, msg, msgID] = mkdir(dirn) ;
if status ~= 1
   warning(['Failed to make dir: ', dirn]) 
   disp(msg)
else
    if ~isempty(msg)
        disp(msg)
    end
    disp(['Made folder: ',dirn])
end


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


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% zip XML
 
xmld = char(get(handles.text10,'String')) ;
zipfn = [xmld,'.zip'] ;

disp(['Creating zip file: ',zipfn,' from: ',xmld])
set(handles.text12,'String','ZIPPING ...')
drawnow
fzipped = zip( zipfn, xmld) ;
disp(['zipped ',num2str(length(fzipped)),' files.'])
set(handles.text12,'String','zip finished')




function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


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


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Browse for containing folder 
stdname = char(get(handles.edit3,'String')) ;
disp(['Select containing folder for ',stdname,' and sub-folders'])
contain_folder = pref_uigetdir('reclaim','contain_folder') ;

set(handles.edit4,'String',contain_folder)


% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Copy XML and REC from where anonymised to new sub-folder

src = char(get(handles.edit1,'String')) ;
dest = char(get(handles.text10,'String')) ;
dxml = dir(fullfile(src,'*.xml')) ;
drec = dir(fullfile(src,'*.rec')) ;
if length(dxml) ~= length(drec)
    warning(['Should be same number of xml and rec files in ',src])
else
    disp(['Starting copy'])
    [status, msg] = copyfile(src,dest) ;
    if status == 0 
        warning(msg)
    else
        disp(['Copy finished'])
    end
end
    
function update_listbox(handles)

lbs = get(handles.listbox1,'String') ;
containing = char(get(handles.edit3,'String')) ;
xmld = char(get(handles.text10,'String')) ;
zipfn = [xmld,'.zip'] ;
[fp,xfn,fext] = fileparts(zipfn) ;
dd = char(get(handles.text11,'String')) ;
[fp,dfn,dext] = fileparts(dd) ;


lbs{5} = ['mkdir ', containing] ;
lbs{6} = ['cd ',containing];
lbs{7} = ['lcd ''',get(handles.text13,'String'),'''' ] ;
lbs{8} = ['put ', xfn, '.zip'] ;
lbs{9} = ['put ', dfn, '.zip'] ;

set(handles.listbox1,'String',lbs)


% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% zip DICOM

dd = get(handles.text11,'String') ;
zipfn = [dd,'.zip'] ;

disp(['Creating DICOM zip file: ',zipfn,' from: ',dd])
set(handles.text12,'String','ZIPPING ...')
drawnow
fzipped = zip( zipfn, dd) ;
disp(['zipped ',num2str(length(fzipped)),' files.'])
set(handles.text12,'String','zip finished')



% --- Executes on key press with focus on pushbutton11 and none of its controls.
function pushbutton11_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
