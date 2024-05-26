function dwrite(data, varargin) 
% DWRITE DICOM write !! Not for Clinical Use !! Superceded by writeDicom
%
% dwrite(data, param, value, ...)
%
% data  [ny nx nf]
%
% This was a quick fix function - better to amend writeDicom where
% possible.
%
% parameter value pairs
% 
% 'fnstem'
% 'geom'  geom structure
% 'SeriesDescription' defaults to 'MATLAB dwrite'
% 'RefFile' Reference file for Patient and Study Information
% 'FrameOfReferenceUID'  FrameOfReferenceUID (defaults to new)
% 'ImageType' defaults to 'DERIVED\SECONDARY\MATLAB'
% 'SeriesNumber' defaults to 4343 
% 'DataRange' {'full'} 'positive'  
%
% !! Comments Below Appear Incorrect!!
% For 'new', needs geom structure with IOP,IPP,etc
% data last dimension must be same as geom size
%
% For 'copy', needs a dinfo and loc from which to access information in
% dinfo struct.
% 
% FrameOfReferenceUID 
%  Set to new value, unless: 
%   geometry 'copy' dinfo(locs) has exactly one value
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also writeDicom

hw  = warndlg({'DICOM files not for clinical use.' ; ...
    'No check here for correct geometry.' ; ...
    'Output will overwite existing files with same name'} ,...
    'Not for clinical use') ;
uiwait(hw)


SeriesDescription = 'MATLAB dwrite' ;
ImageType = 'DERIVED\SECONDARY\MATLAB' ;
RefFile = '' ;
FrameOfReferenceUID = dicomuid ;
scale_method = 'full' ;
SeriesNumber = 4343 ;
DataRange = 'full' ;

for ipv = 1: 2 :length(varargin)
    switch varargin{ipv}
        case 'SeriesDescription'
            SeriesDescription = varargin{ipv+1} ;
        case 'fnstem'
            fnstem = varargin{ipv+1} ;
        case 'geom'
            geom = varargin{ipv+1} ;
        case 'RefFile'
            RefFile = varargin{ipv+1} ;
        case 'FrameOfReferenceUID'
            FrameOfReferenceUID = varargin{ipv+1} ;
        case 'ImageType'
            ImageType = varargin{ipv+1} ;
        case 'SeriesNumber'
            SeriesNumber = varargin{ipv+1} ;
        case 'DataRange'
            DataRange = varargin{ipv+1} ;
        otherwise
            warning(['Unknown parameter: ',varargin{ipv}])
    end
end

% Checks on data sizes etc
sz_data = size(data) ;

if length(sz_data) > 3
    error(['Only 2D or 3D data currently supported'])
end

if length(sz_data) == 2
    nf = 1 ;
else
    nf = sz_data(end) ;
end

if length(geom) ~= nf
    error(['geom and data size must agree'])
end

% Patient and Study Data to copy over (Series data handled later)
if exist(RefFile,'file')
    tags = { ...
        'SOPClassUID'
        'StudyDate'
        'StudyTime'
        'StudyID'
        'StudyInstanceUID'
        'Modality'
        'Manufacturer'
        'ManufacturerModelName'
        'PatientName'
        'PatientID'
        'PatientBirthDate'
        'PatientSex'
        'OtherPatientID'
        'PatientWeight'
        'BodyPartExamined' };
    
    dout = dextract_fixed(RefFile, tags) ;
    if isfield(dout,'SOPClassUID')
        if strcmp(dout.SOPClassUID, '1.2.840.10008.5.1.4.1.1.4.1')
            dout.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4' ;
            disp(['RefFile was Enhanced MR, output will be single frame per file']) 
        end
    end
else
    dout = [] ;
end

% Series information
dout.SeriesDescription = SeriesDescription ;
dout.SeriesInstanceUID = dicomuid ; 
dout.FrameOfReferenceUID = FrameOfReferenceUID ; 
dout.ImageType = ImageType ;
dout.SeriesNumber = SeriesNumber ;

folder_name = pref_uigetdir('writeDicom','save_dir') ;
switch DataRange
    case 'full'
        drange = [min(data(:)) max(data(:))] ;
    case 'positive'
        drange = [max([0 min(data(:))])  max(data(:))] ;
end

for iframe = 1:nf
    dout.Width = geom(iframe).Width ;
    dout.Height = geom(iframe).Height ;
    dout.PixelSpacing = [geom(iframe).PixelSpacing_HW(1) geom(iframe).PixelSpacing_HW(2)];
    dout.ImageOrientationPatient = geom(iframe).IOP ;
    dout.ImagePositionPatient = geom(iframe).IPP ;
    dout.SliceThickness = geom(iframe).SliceThickness ;
    
    fn_out = fullfile(folder_name,[fnstem,'_',num2str(iframe,'%03u'),'.dcm']) ;
    
    [pdata, rs, ri] = dscale(data(:,:,iframe),'uint16', drange, scale_method) ;
    dout.RescaleSlope = rs ;
    dout.RescaleIntercept = ri ;
    
    status = dicomwrite(pdata, fn_out, dout ) ;
    if rs~=1  || ri~=0
        % dicomwrite doesn't write these out in Create Mode
        % need to overwrite in Copy mode!
        tinfo = dicominfo(fn_out) ;
        delete(fn_out) ;
        tinfo.RescaleSlope = rs ;
        tinfo.RescaleIntercept = ri ;

        status = dicomwrite(pdata, fn_out, tinfo, ...
            'CreateMode','Copy') ;
    end
    
end

disp(['Written ',num2str(nf),' files to ',folder_name])


end % function dwrite



%----------------------------------------------------------
function info_fixed = dextract_fixed(RefFile, tags) 
%
% Currently uses only first file

dfull = dicominfo(RefFile) ;

ntags = length(tags) ;

for itag = 1:ntags
    if isfield(dfull,tags{itag})
        info_fixed.(tags{itag}) = dfull.(tags{itag}) ;
    end
end

end % dextract_fixed


    
    
