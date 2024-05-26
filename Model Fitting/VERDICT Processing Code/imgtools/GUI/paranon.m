function varargout = paranon(varargin)
% PARANON Anonymise and rename Philips PAR/REC files
% Usage:  paranon   - brings up GUI.
%
% ! Hacked a bit after finding no component with the edit_patientname tag.
% Now seems OK (14 April, 2022)
%
%   Close all applications that use the PAR or REC files to be anonymised.
%   Browse to folder containing PAR & REC files of one (and only one) subject.
%   Enter suitable name for anonymisation -  a name based on the time
%      and date will have been suggested.
%   Examination Date /Time will be set to 1 Jan unless you edit or copy
%      original (Keeping exact date aids re-identification but may compromise
%      anonymity)
%   Modify 'Dataset name' ritten into the PAR file. 
%   Modify the partial file stem for the PAR and REC files. Note changing
%   this will prevent the GUI being able to rename these files again unless
%   the change matches the patient name.
%
%   Selecting Apply will;
%     For each PAR file, a temporary copy will be written with updated
%     information (i.e. new patient name, exam date and dataset name).
%     The original is then overwritten with the temporary copy.
%     The new PAR and the existing REC files are renamed if the originals
%     contain the patient name in the file name,
%
% PAR/REC files contain the following information that could enable subject
% identification:
%  * Patient Name  - this is stored in the PAR file.
%  * Examination Date/time - stored in the PAR file. Re-identification may be
%  possible with this information and access to PACs. 
%  * The PAR and REC file names may contain the patient name.
%  * A comment called 'Dataset name' contains the original path name which
%  might include the patient name.
%
% David Atkinson   D.Atkinson@ucl.ac.uk
%

% PARANON MATLAB code for paranon.fig
%      PARANON, by itself, creates a new PARANON or raises the existing
%      singleton*.
%
%      H = PARANON returns the handle to a new PARANON or the handle to
%      the existing singleton*.
%
%      PARANON('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PARANON.M with the given input arguments.
%
%      PARANON('Property','Value',...) creates a new PARANON or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before paranon_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to paranon_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help paranon

% Last Modified by GUIDE v2.5 15-Apr-2022 11:10:10

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @paranon_OpeningFcn, ...
                   'gui_OutputFcn',  @paranon_OutputFcn, ...
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

% --- Executes just before paranon is made visible.
function paranon_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to paranon (see VARARGIN)

% Choose default command line output for paranon
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes paranon wait for user response (see UIRESUME)
% uiwait(handles.figure1);

setanon(handles) ;

end

function setanon(handles)

UD = get(handles.figure1,'UserData') ;
if isempty(UD)
    UD = [] ; % Can return empty graphics object, but want regular struct
end
[Y, M, D, H, MN, S] = datevec(now) ;
str = sprintf('%4d%02d%02d%02d%02d%02d',Y,M,D,H,MN,round(S));

set(handles.edit_rpn,'String',['anon',str]) ;
set(handles.edit_filestem,'String',['anon',str]) ;
UD.isfset = 0 ;
set(handles.figure1,'UserData',UD) ;

end

% --- Outputs from this function are returned to the command line.
function varargout = paranon_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

end


function edit_directory_Callback(hObject, eventdata, handles)
% hObject    handle to edit_directory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_directory as text
%        str2double(get(hObject,'String')) returns contents of edit_directory as a double

update_namelist(handles) ;

end

% --- Executes during object creation, after setting all properties.
function edit_directory_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_directory (see GCBO)
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
folder_name = pref_uigetdir('paranon','folder','','Select PAR/REC folder') ;

if isnumeric(folder_name) || ~exist(folder_name,'dir')
    folder_name = 'Select valid folder' ;
end

set(handles.edit_directory,'String',folder_name)

update_namelist(handles) ;

end



function edit_rpn_Callback(hObject, eventdata, handles)
% hObject    handle to edit_rpn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_rpn as text
%        str2double(get(hObject,'String')) returns contents of edit_rpn as a double

update_filestem(handles) ;
update_stored_filenames(handles) ;

end


% --- Executes during object creation, after setting all properties.
function edit_rpn_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_rpn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end

% --- Executes on button press in pushbutton_apply.
function pushbutton_apply_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_apply (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(hObject,'Enable','off') ;
nover = overwrite_par(handles) ;
nmove = move_files(handles) ;

% leaving Enable off will force reread of directory and setting of
% parameters

set(handles.text_nomodified,'String',[num2str(nover),' PAR files were modfied and ', ...
    num2str(nmove),' files renamed.'])

end

% --- Executes on button press in pushbutton_cancel.
function pushbutton_cancel_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


close(handles.figure1)

end


function edit_filestem_Callback(hObject, eventdata, handles)
% hObject    handle to edit_filestem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_filestem as text
%        str2double(get(hObject,'String')) returns contents of edit_filestem as a double

str = get(hObject,'String') ;
loc = strfind(str,' ') ;
if ~isempty(loc)
    set(handles.text_msg,'String','Replacing spaces in File Stem with _') ;
    str = space2under(str) ;
    set(hObject,'String',str);
end
update_stored_filenames(handles) ;

end


% --- Executes during object creation, after setting all properties.
function edit_filestem_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_filestem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


function edit_examdate_Callback(hObject, eventdata, handles)
% hObject    handle to edit_examdate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_examdate as text
%        str2double(get(hObject,'String')) returns contents of edit_examdate as a double

set(handles.text_msg,'String','')
end

% --- Executes during object creation, after setting all properties.
function edit_examdate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_examdate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


function update_namelist(handles) 
  UD = get(handles.figure1,'UserData');
  
  folder = get(handles.edit_directory,'String') ;
  if ~exist(folder,'dir')
      msg = 'Folder does not exist';
      set(handles.text_msg,'String',msg) ;
      set(handles.pushbutton_apply,'Enable','off')
      return
  end
  
  dpar = dir(fullfile(folder,'*.PAR')) ;
  drec = dir(fullfile(folder,'*.REC')) ;

  dremove = [];

  for idpar = 1:length(dpar)
      if dpar(idpar).name(1:2)=='._'
          % Remove Mac(?) ._*.PAR or REC files
          dremove = [dremove idpar];
      end
  end
  dpar(dremove) = [] ;

  
  if isempty(dpar)
      msg = 'No PAR files found' ;
      set(handles.text_msg,'String',msg) ;
      set(handles.pushbutton_apply,'Enable','off')
      return
  end
  
  if length(drec) < length(dpar)
      msg = ['Expected PAR/REC pairs. ',num2str(length(drec)),' REC and ',...
          num2str(length(dpar)),' PAR'];
      set(handles.text_msg,'String',msg) ;
      return
  end
  
  for ipar = 1:length(dpar)
      parv = readparv(fullfile(folder,dpar(ipar).name), handles) ;
      p{ipar} = parv.PatientName ;
  end
  
  pn = unique(p) ; % patient name in tag
  if length(pn) ~= 1
      msg = 'Require exactly 1 patient name in files' ;
      set(handles.text_msg,'String',msg) ;
  end
  
  pfn = space2under(pn) ; 
  set(handles.text_filestem,'String',pfn)
  
  datan = parv.DatasetName ;
  loc = strfind(datan,pfn) ;
  
  new_pn = get(handles.edit_rpn,'String') ;
  new_pfn = space2under(new_pn) ;
  dnew_par = generate_newfilenames(dpar, pfn, new_pfn ) ;
  dnew_rec = generate_newfilenames(drec, pfn, new_pfn ) ;
  
  info_str = [num2str(length(dpar)),' will be modified. ', ...
      num2str(sum([dnew_par.move])),' PAR ',...
      num2str(sum([dnew_rec.move])),' REC will be renamed'] ;
  set(handles.text_nomodified,'String',info_str) ;
  UD.dpar = dnew_par ;
  UD.drec = dnew_rec ;
  UD.isfset = 1 ;
  set(handles.figure1,'UserData',UD)
  
  % Currently not doing multiple files
  
  DatasetNamePath = fileparts(parv.DatasetName) ;
  
  set(handles.text_patientname,'String',parv.PatientName) ;
  set(handles.text_dataset,'String',DatasetNamePath) ;
  %set(handles.edit_dataset,'String',DatasetNamePath) ;
  set(handles.edit_dataset,'String',' ') ;
  set(handles.text_suggestion,'String',DatasetNamePath) ;
  set(handles.text_examdate,'String',parv.ExamDate) ;
  set(handles.edit_examdate,'String',parv.ReplaceDate) ;
  
  set(handles.text_msg,'String','') ;
  set(handles.pushbutton_apply,'Enable','on') ;  
end

function update_stored_filenames(handles) 
UD = get(handles.figure1,'UserData');

if UD.isfset 
    folder = get(handles.edit_directory,'String') ;
    dpar = dir(fullfile(folder,'*.PAR')) ;
    drec = dir(fullfile(folder,'*.REC')) ;
    
    old_pfn = get(handles.text_filestem,'String') ;
    new_pfn = get(handles.edit_filestem,'String') ;
    
    dnew_par = generate_newfilenames(dpar, old_pfn, new_pfn ) ;
    dnew_rec = generate_newfilenames(drec, old_pfn, new_pfn ) ;
    
    info_str = [num2str(length(dpar)),' will be modified. ', ...
      num2str(sum([dnew_par.move])),' PAR ',...
      num2str(sum([dnew_rec.move])),' REC will be renamed'] ;
  
    set(handles.text_nomodified,'String',info_str) ;
    UD.dpar = dnew_par ;
    UD.drec = dnew_rec ;
    set(handles.figure1,'UserData',UD)
end

end


function nover = overwrite_par(handles)
% OVERWRITE_PAR
% read in and copy to temp file
% overwrite original with temp
% delete temp

% get new_pn from edit_
% get new_ex from edit_
% get new_path from edit_
%
% read in dir *.PAR, select those with patient name 
%   loop over files with patient name
%   for new path, use move filename to work out naming.   

nover = 0 ;

new_pn = get(handles.edit_rpn,'String') ;
pfn = get(handles.text_filestem,'String') ;
new_pfn = get(handles.edit_filestem,'String') ;
new_ex = get(handles.edit_examdate,'String') ;
new_dnpath = get(handles.edit_dataset,'String') ;

direc = get(handles.edit_directory,'String') ;
dpar = dir(fullfile(direc,'*.PAR')) ;

for ipar = 1:length(dpar)
    parfn = fullfile(direc,dpar(ipar).name) ;
    
    % temporary file for new PAR file before overwriting existing
    tempfn = tempname ;
    [fid_temp, mesg] = fopen(tempfn, 'w');
    if fid_temp == -1
        set(handles.text_msg,'String',mesg) ;
        set(handles.pushbutton_apply,'Enable','off')
        return
    end
    
    [fid_par, mesg] = fopen(parfn, 'r');
    if fid_par == -1
        set(handles.text_msg,'String',mesg) ;
        set(handles.pushbutton_apply,'Enable','off')
        return
    end
    
    tline = fgets(fid_par);
    while ischar(tline)
        match_PN = strfind(tline, 'Patient name');
        if ~isempty(match_PN)
            % note the '.' is not in the line below (compared to parread)
            [tcolon] = textscan(tline,'%[^:]:%s','whitespace', '\b\t');
            tline = sprintf('%s\r\n',[tcolon{1}{1},':   ',new_pn]) ;
        end
        
        match_EX = strfind(tline, 'Examination date/time');
        if ~isempty(match_EX)
            [tcolon] = textscan(tline,'%[^:]:%s','whitespace', '\b\t');
            tline = sprintf('%s\r\n',[tcolon{1}{1},':   ',new_ex]) ;
        end
        
        % Dataset name is the filestem of the original files written by the
        % scanner without a .PAR or .REC extension. If the patient name was
        % in the original files, it will be in here too. Also, if a folder
        % was created with the patient name, that will also be here so the
        % GUI user has the option to change the path. The patient name is
        % removed automatically and replaced with that used for the
        % filenames. 
        match_DN = strfind(tline, 'Dataset name');
        if ~isempty(match_DN)
            [tcolon] = textscan(tline,'%[^:]:%s','whitespace', '\b\t');
            olddn.name = tcolon{2}{1} ;
            newdn = generate_newfilenames(olddn, pfn, new_pfn ) ;
            [pth, newdfn] = fileparts(newdn.new_name) ;
            
            new_dn = fullfile(new_dnpath, newdfn) ;
            tline = sprintf('%s\r\n',[tcolon{1}{1},': ',new_dn]) ;
        end
        
        fprintf(fid_temp,'%s',tline) ;
        
        tline = fgets(fid_par);
    end
    
    status1 = fclose(fid_par) ;
    status2 = fclose(fid_temp) ;
    
    if status1 == -1
        set(handles.text_msg,'String','Unable to close par file') ;
    end
    
    if status2 == -1
        set(handles.text_msg,'String','Unable to close temp file') ;
    end
    
    [s,mess,messid] = movefile(tempfn, parfn) ;
    if s~= 1
        errstr = [mess,'  ',messid] ;
        set(handles.text_msg,'String',errstr)
    else
        nover = nover + 1 ;
        disp(['Updated file: ',parfn])
    end
    
end % loop over par files in directory
end

function nmove = move_files(handles)
nmove = 0 ;
folder = get(handles.edit_directory,'String') ;
UD = get(handles.figure1,'UserData') ;

for id = [1 2]
    if id ==1 ; dstruc = UD.dpar ; else dstruc = UD.drec; end
    
    for ifile = 1:length(dstruc)
        if isfield(dstruc,'move') && dstruc(ifile).move == 1
            [s,mess,messid]=movefile(fullfile(folder,dstruc(ifile).name), ...
                fullfile(folder,dstruc(ifile).new_name))  ;
            if s~= 1
                disp(['Cannot move ',fullfile(folder,dstruc(ifile).name),...
                    ' to ',fullfile(folder,dstruc(ifile).new_name) ])
                errstr = [mess,'  ',messid] ;
                set(handles.text_msg,'String',errstr)
            else
                nmove = nmove + 1;
                disp(['Moved ',dstruc(ifile).name,' to  ',...
                    dstruc(ifile).new_name])
            end
        end
    end % ifile
end % dstruc
end

function dnew = generate_newfilenames(dold, pfn, new_pfn ) 
dnew = dold ;
for ifile = 1:length(dold) 
   currfn = dold(ifile).name ;
   loc = strfind(currfn,pfn) ;
   if length(loc) == 1
       newstr = [currfn(1:loc-1) new_pfn currfn(loc+length(pfn):end)];
       dnew(ifile).move = true ;
       disp(['Old: ',currfn,' will become: ',newstr])
   else
       newstr = currfn ;
       dnew(ifile).move = false ;
   end
   
   dnew(ifile).new_name = newstr ;
end

end

function update_filestem(handles)
anon_pn = get(handles.edit_rpn,'String') ;
fstem = space2under(anon_pn) ;
set(handles.edit_filestem,'String',fstem)
update_stored_filenames(handles) ;

end

function parv = readparv(fn, handles)
[fid, msg] = fopen(fn,'rt') ;
if fid==-1
    set(handles.text_msg,'String',msg) ;
end

tline = fgetl(fid);
while ischar(tline)
   match_PN = strfind(tline, 'Patient name');
   if ~isempty(match_PN)
       [tcolon] = textscan(tline,'.%[^:]:%s','whitespace', '\b\t');
       if ~isempty(tcolon{2})
         parv.PatientName = deblank(tcolon{2}{1}) ;
       else
           parv.PatientName = '';
       end
   end
   
   match_EX = strfind(tline, 'Examination date/time');
   if ~isempty(match_EX)
       [tcolon] = textscan(tline,'.%[^:]:%s','whitespace', '\b\t');
       if ~isempty(tcolon{2})
         parv.ExamDate = deblank(tcolon{2}{1}) ;
       else
          parv.ExamDate = '    ';
       end
       parv.ReplaceDate = [parv.ExamDate(1:4),'.01.01 / 00:00:00'];
   end
   
   match_DN = strfind(tline,'Dataset name') ;
   if ~isempty(match_DN)
       [tcolon] = textscan(tline,'#%[^:]:%s','whitespace', '\b\t');
       if length(tcolon{2})>0
          parv.DatasetName = deblank(tcolon{2}{1}) ;
       else
          parv.DatasetName = '';
       end
   end
   
   
   tline = fgetl(fid);
end
fclose(fid);
end

function pfn = space2under(pn)
% SPACE2UNDER Replace spaces with underscore in string

if iscell(pn)
    pn = pn{1} ;
end

loc = strfind(pn,' ') ;
pfn = pn ;
pfn(loc) = '_' ;


end

function edit_dataset_Callback(hObject, eventdata, handles)
% hObject    handle to edit_dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_dataset as text
%        str2double(get(hObject,'String')) returns contents of edit_dataset as a double

set(handles.text_msg,'String','')

end

% --- Executes during object creation, after setting all properties.
function edit_dataset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end


% --- Executes on button press in pushbutton_setanon.
function pushbutton_setanon_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_setanon (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

setanon(handles) ;
end


% --- Executes on button press in pushbutton_copy.
function pushbutton_copy_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_copy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

str = [get(handles.text_patientname,'String'), ' , ', ...
       get(handles.edit_rpn,'String')] ;
   
clipboard('copy',str) ;

set(handles.text_msg,'String','Names on clipboard - paste to desired location')

end


% --- Executes on button press in pushbutton_copydate.
function pushbutton_copydate_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_copydate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.edit_examdate,'String', get(handles.text_examdate,'String')) ;

end


% --- Executes on button press in pushbutton_suggest.
function pushbutton_suggest_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_suggest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.edit_dataset,'String', get(handles.text_suggestion,'String')) ;

end



function edit_patientname_Callback(hObject, eventdata, handles)
% hObject    handle to edit_rpn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_rpn as text
%        str2double(get(hObject,'String')) returns contents of edit_rpn as a double

end



% --- Executes during object creation, after setting all properties.
function edit_patientname_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_rpn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


