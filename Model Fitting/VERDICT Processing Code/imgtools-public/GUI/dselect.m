function varargout = dselect(varargin)
% DSELECT GUI file selection for Enhanced DICOMs and Philips XX* DICOMs
% Provides useful series information (even after anonymisaton).
%
%      filenm = dselect
%      filenm = dselect('Param',value, ...)
%
%      Param can be 'ismodal', 'name', 'xxonly', 'prefname' or 'message'
%
%      'XX' DICOMs are created by Philips MR and contain sequence
%      information.
%
% Copyright, 2019, David Atkinson 
% D.Atkinson@ucl.ac.uk
%
%      DSELECT, by itself, creates a new DSELECT or raises the existing
%      singleton*.
%
%      H = DSELECT returns the handle to a new DSELECT or the handle to
%      the existing singleton*.
%
%      DSELECT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DSELECT.M with the given input arguments.
%
%      DSELECT('Property','Value',...) creates a new DSELECT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before dselect_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to dselect_OpeningFcn via varargin.
%
%      Properties added are 'Name', 'ismodal', 'xxonly'
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help dselect

% Last Modified by GUIDE v2.5 20-Oct-2016 20:08:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @dselect_OpeningFcn, ...
                   'gui_OutputFcn',  @dselect_OutputFcn, ...
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


% --- Executes just before dselect is made visible.
function dselect_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to dselect (see VARARGIN)

% Choose default command line output for dselect
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

ismodal = true ; % default
set(handles.pushbutton_finish,'Visible','on')

UD = get(hObject,'UserData') ;
if isempty(UD) 
    clear UD
end
UD.prefname = 'dir'; % default prefname

for ip = 1:2:length(varargin)
    switch varargin{ip}
        case 'ismodal'
            switch varargin{ip+1}
                case true
                    ismodal = true ;
                    set(handles.pushbutton_finish,'Visible','on')
                case false
                    ismodal = false ;
                    set(handles.pushbutton_finish,'Visible','off')
                otherwise
                    warning(['ismodal value must be true or false'])
            end
            
        case {'name','Name'}
            set(handles.figure1,'Name',varargin{ip+1})
            
        case {'xxonly'}
            if varargin{ip+1}
                set(handles.listbox2, 'Value',[3 6])
            end
        case 'message'
            set(handles.text3,'String', varargin{ip+1})
        case 'prefname'
            UD.prefname = varargin{ip+1} ;
            
        otherwise
            warning(['Unknown parameter: ',varargin{ip}])
    end
end

UD.ismodal = ismodal ;
set(hObject,'UserData',UD) 
                
% UIWAIT makes dselect wait for user response (see UIRESUME)
if ismodal
  uiwait(handles.figure1);
end


% --- Outputs from this function are returned to the command line.
function varargout = dselect_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
if ~isempty(handles)
  varargout{1} = handles.output;
else
    varargout{1} =[];
    return
end
if nargout > 0
    UD = get(hObject,'UserData') ;
    lval = get(handles.listbox1,'Value') ;
    if isfield(UD,'flist') && ~isempty(UD.flist)
        varargout{1} = {UD.flist{lval}} ;
    else
        varargout{1} = [] ;
    end
else
    % output to screen for possible cut and paste
    UD = get(hObject,'UserData') ;
    lval = get(handles.listbox1,'Value') ;
    disp({UD.flist{lval}}')
end
set(handles.text3,'String','')




% --- Executes on button press in pushbutton_close.
function pushbutton_close_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_close (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(handles.figure1)


function edit1_folder_Callback(hObject, eventdata, handles)
% hObject    handle to edit1_folder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1_folder as text
%        str2double(get(hObject,'String')) returns contents of edit1_folder as a double
update_dir_string(handles) ;

% --- Executes during object creation, after setting all properties.
function edit1_folder_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1_folder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton2_browse.
function pushbutton2_browse_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

UD = get(handles.output,'UserData') ;
prefname = UD.prefname ;

if ispref('dselect',prefname)
    defdir = getpref('dselect',prefname);
else
    defdir = [] ;
end

folder_name = uigetdir(defdir,'Select Multi-Frame DICOM folder') ;
if folder_name == 0
    folder_string = 'Enter valid folder' ;
else
    folder_string = folder_name ;
    setpref('dselect',prefname,folder_name)
    UD.prefname = prefname ;
    set(handles.output,'UserData',UD)
end

set(handles.edit1_folder,'String',folder_string)
update_dir_string(handles) ;


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


function update_namelist(handles) 
% Attributes in order they will be listed.
% A non-existet attribute will force reading of whole header.
% Attributes should be strings, if not modify lstr loop below.
% attrn = {'SeriesNumber','SeriesDescription','PatientName','PatientID'} ;
% Note that SeriesDescription may be missing from Philips scanner
% anonymised files

% attrn = {'SeriesNumber','SeriesDescription'} ;
attrn = {'SeriesNumber','ProtocolName', 'SeriesDescription'} ;

if get(handles.checkbox4,'Value') == 1
    opfn = true ;
else
    opfn = false ;
end

set(handles.listbox1,'String','Updating ... (Doing DIR)')
drawnow update
folder = get(handles.edit1_folder,'String') ;
dir_str = get(handles.edit3_dir_string,'String') ;

d = dir(dir_str) ;
nf = length(d) ;
ilbx = 0 ;
lbx = cell(1) ;

lb1_str = ['Updating (',num2str(nf),' files) ....'] ;
set(handles.listbox1,'String',lb1_str)
drawnow update
EMR = '1.2.840.10008.5.1.4.1.1.4.1' ;
RAW = '1.2.840.10008.5.1.4.1.1.66' ;
SCC = '1.2.840.10008.5.1.4.1.1.7' ;
GPS = '1.2.840.10008.5.1.4.1.1.11.1' ;
PDS = '1.3.46.670589.11.0.0.12.2' ; % Philips Private Data Storage (single frame DICOM)
PEC = '1.3.46.670589.11.0.0.12.4' ; % Philips ExamCard
SR  = '1.2.840.10008.5.1.4.1.1.88.11' ; % Structured Report (for OsiriX ROIs)
SOPClassUIDs = {'all', EMR, RAW, SCC, GPS, PDS, PEC, SR } ;

vs = get(handles.listbox2,'Value') ;
SOPClass = {SOPClassUIDs{vs}} ;

if strcmp(SOPClass, SR) % Add attributes for SR
  attrn = {'SeriesNumber','ProtocolName', 'SeriesDescription','PatientName', ...
      'StudyDate', 'StudyTime', ...
      'ContentDate', 'ContentTime', 'ReferencedFrameNumber'} ;  
end

for ifile = 1:nf
    if ~d(ifile).isdir && d(ifile).bytes>132
        dinfo = dfastinfo(fullfile(folder,d(ifile).name), attrn, SOPClass) ;
        if ~isempty(dinfo)
            lstr = '' ;
            
            for iattr = 1:length(attrn)
                if isfield(dinfo, attrn{iattr})
                    if strcmp(attrn{iattr},'SeriesNumber')
                        sopstr = '    ';
                        if isfield(dinfo,'SOPClassUID')
                            switch dinfo.SOPClassUID
                                case EMR
                                    sopstr = ' EMR' ;
                                case RAW
                                    sopstr = ' RAW' ;
                                case SCC
                                    sopstr = ' SCC' ;
                                case GPS
                                    sopstr = ' GPS' ;
                                case PDS
                                    sopstr = ' PDS' ;
                                case PEC
                                    sopstr = ' PEC' ;
                                case SR
                                    sopstr = ' SR';
                                otherwise
                                    sopstr = [' ',dinfo.SOPClassUID] ;
                            end
                        end
                        lstr = [lstr,sprintf('%04s',dinfo.SeriesNumber),sopstr,' | '];
                    else
                        lstr = [lstr, dinfo.(attrn{iattr}),' '];
                    end
                end
                
            end

            if opfn
                lstr = [lstr, ' [',d(ifile).name,'] '] ;
            end
            
            ilbx = ilbx + 1 ;
            flist{ilbx} = fullfile(folder,d(ifile).name) ;
            lbx{ilbx} = lstr ;
            
            set(handles.listbox1,'String',{lbx{:},['UPDATING ... (Done ',d(ifile).name,')']})
            set(handles.listbox1,'Value',ilbx+1)
            drawnow update
        end
    end
    
end

UD = get(handles.output,'UserData') ;
if ilbx == 0
    set(handles.listbox1,'String','NO MATCHING DICOM FILES FOUND.')
    UD.flist = [] ;
else
    
    [sort_lbx, idx] = sort(lbx) ;
    flist = flist(idx) ;
    
    
    UD.flist = flist ;
    
    set(handles.listbox1,'String',sort_lbx)
    set(handles.listbox1,'Value', 1)
end
set(handles.output,'UserData', UD)

function update_dir_string(handles)
folder = get(handles.edit1_folder,'String') ;
reg_exp = get(handles.edit2,'String') ;
dir_string = fullfile(folder, reg_exp) ;

set(handles.edit3_dir_string, 'String', dir_string)

d = dir(dir_string) ;
nd = sum([d.isdir]) ;
set(handles.listbox1,'Value',1)
set(handles.listbox1,'String',['Entries: ',num2str(length(d)),' (',num2str(nd),' folders)'])

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


% --- Executes on button press in pushbutton_finish.
function pushbutton_finish_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_finish (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

UD = get(handles.output,'UserData') ;
if UD.ismodal == true
    uiresume(handles.figure1) ;
end

set(hObject,'Visible','off')


% --- Executes on button press in checkbox4.
function checkbox4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox4



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double
update_dir_string(handles)

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



function edit3_dir_string_Callback(hObject, eventdata, handles)
% hObject    handle to edit3_dir_string (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3_dir_string as text
%        str2double(get(hObject,'String')) returns contents of edit3_dir_string as a double


% --- Executes during object creation, after setting all properties.
function edit3_dir_string_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3_dir_string (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton4_dir.
function pushbutton4_dir_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4_dir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

update_namelist(handles) ;
