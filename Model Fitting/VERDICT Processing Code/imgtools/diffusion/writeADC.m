function writeADC(ADC, dinfo, locs, fnstem)
% writeADC  Write DICOM ADC !!! Replaced by writeDicom
%  writeADC(ADC, dinfo, locs, fnstem)
%
% !! NOT FOR CLINICAL USE !! (No geometry, frame of ref, or patient checks)
%
% ADC:  [ny nx nz]  Expect that after multiplying by 1e6, ADC will have 
%       values in range 0-4000.
% dinfo:  output from datparse
% locs: [nz nbv]  locations in dinfo of slices and b-values
% fnstem: Filename stem for output filenames, defaults to 'ADC'.
%
% Example use
% ===========
%  dinfo = datparse ;
%  [vol, mat, locs] = d2mat(dinfo,{'slice','bv','series'},'series',13,'op','fp') ;
%  [ADC,S0] = calcADC(vol,mat.bvVec) ;
%  writeADC(ADC, dinfo, locs, 'testADC')
%
% 
% D.Atkinson@ucl.ac.uk
%
% See also WRITEDICOM D2MAT calcADC DATPARSE
%

SeriesDescription = 'Calculated ADC' ;
ImageType = 'DERIVED\SECONDARY\ADC' ;  %?? Not sure of standard here

scale = 1e6 ; % ADC scaling factor
ADCupper = 4000 ;  % ADC values over this are written as 4000.
WindowWidth = 1999 ;
WindowCenter = 1000 ;

if nargin < 4
    fnstem = 'ADC' ;
end

nslice = size(ADC,3) ;
if size(locs,1) ~= nslice
    error(['Slices in ADC and locs do not match'])
end

% Check ADCs in range 0 to ADCupper
ADC(ADC<0) = 0 ;

ADC = ADC * scale ;
loc_large = find(ADC > ADCupper) ;

hw  = warndlg({'DICOM files not for clinical use.' ; ...
    'No check here for correct geometry.' ; ...
    'Output will overwite existing files with same name'} ,...
    'Not for clinical use') ;
uiwait(hw)


if ispref('writeADC','save_dir')
    defdir = getpref('writeADC','save_dir');
else
    defdir = [] ;
end

[folder_name] = uigetdir(defdir,'Select folder for output') ;


if folder_name == 0
    warning(['No folder specified'])
    return
else
    setpref('writeADC','save_dir',folder_name)
end



if ~isempty(loc_large)
    warning(['Clipping ADC values over ',num2str(ADCupper)])
    ADC(loc_large) = ADCupper; 
end

ADC = uint16(round(ADC)) ;

new_dicomuid = dicomuid ; % new fake series UID.

for islice = 1:nslice
    b0fn = dinfo(locs(islice,1)).Filename ;
    
    dinfull = dicominfo(b0fn) ;
    
    dinfo_out = dinfull ;
    
    dinfo_out.SeriesInstanceUID = new_dicomuid ; 
    dinfo_out.ImageType = ImageType ; 
    dinfo_out.SeriesDescription = SeriesDescription ;
    dinfo_out.WindowWidth = WindowWidth ;
    dinfo_out.WindowCenter = WindowCenter ;
    
    fn_out = fullfile(folder_name,[fnstem,'_',num2str(islice,'%03u'),'.dcm']) ;
    status = dicomwrite(ADC(:,:,islice), fn_out, dinfo_out ) ;
end

disp(['Written ',num2str(nslice),' files to folder: ',folder_name])

