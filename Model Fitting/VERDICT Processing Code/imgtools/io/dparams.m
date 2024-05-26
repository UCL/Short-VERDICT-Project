function dparams
% DPARAMS  DICOM parameter summary.
% Lists summary of DICOM parameters to screen.
%
% Only tested on SingleFrame DICOMs. Now goes recursively down a folder tree.
% Checks that all scans are from same patient name. Does not check ID and 
% no further consistency checks. Values of TR, TE etc are taken from 
% first image in a series.
% 
%
% Example:
%  dparams ; % Calls GUI, lists summary of parameters to screen.
%
% D.Atkinson@ucl.ac.uk

% SOPClassUIDs only needed if using dfastinfo instead of dicominfo 
% (which turns out not to be so fast anymore ...).
%
% EMR = '1.2.840.10008.5.1.4.1.1.4.1' ; % SOPClassUID
% MR3 = '1.2.840.10008.5.1.4.1.1.4' ;

% Attributes to be saved or checked
attrn = {'ProtocolName', ...
    'PatientName', ...
    'StudyDate', ...
    'StudyTime', ...
    'SeriesNumber', ...
    'SeriesDescription', ...
    'SeriesTime', ...
    'SeriesDate', ...
    'ImageComments', ...
    'SequenceName', ...
    'SliceThickness', ...
    'SpacingBetweenSlices', ...
    'PixelBandwidth', ...
    'EchoTrainLength', ...
    'RepetitionTime', ...
    'EchoTime', ...
    'FlipAngle', ...
    'NumberOfPhaseEncodingSteps', ...
    'NumberOfAverages', ...
    'PixelSpacing', ...
    'Width',  ...
    'Height', ...
    'InversionTime', ...
    'ScanningSequence', ...
    'AcquisitionMatrix',...
    'InPlanePhaseEncodingDirection',...
    'VariableFlipAngleFlag',...
    'SequenceVariant', ...
    'ScanOptions',...
    } ;


dcmdir = pref_uigetdir('dparams','dir',[],'Select Single Frame DICOM folder') ;

fns = dflist(dcmdir) ;  % generates list of DICOM filenames


nf = length(fns) ;
hw = waitbar(0,['Parsing ',num2str(nf),' DICOM files']) ;

for ifl = 1:nf
    hw = waitbar(ifl/nf,hw) ;
    dcmfile = fns{ifl} ;
    
    % dinfo = dfastinfo(dcmfile, attrn, MR3) ;
    dinfo = dicominfo(dcmfile);
    
    for iattr = 1:length(attrn)
        this_attr = attrn{iattr} ;
        if isfield(dinfo,this_attr)
            dout(ifl).(this_attr) = dinfo.(this_attr) ;
            % disp([dcmfile,' ',num2str(dinfo.SeriesNumber)])
        end
    end
end

close(hw)

if nf == 0 
    disp(['No DICOM files found.'])
    return
end

% Check just one patient in data
pns = [dout.PatientName] ;

[pn, ian] = unique({pns.FamilyName}) ;
if length(pn) > 1
    warning(['More than one patient name in data:'])
    fnall = {pns.FamilyName} ;
    
    for ireppn = 1:length(pn)
        k=strfind(fnall,pn{ireppn}) ;
        disp([pn{ireppn},' ',num2str(sum([k{:}])),' times'])
        dout(ian(ireppn))
    end
end


% Find Series numbers
snall = [dout.SeriesNumber] ;
[sn, ia] = unique(snall) ;

disp(' ')
hdr_str = '';

% StudyDate not in Leuven DICOMs - use a series date
if isfield(dout(1), 'StudyDate') && isfield(dout(1), 'StudyTime')
    hdr_str = [hdr_str ,datestr(datenum([dout(1).StudyDate,' ',dout(1).StudyTime],'yyyymmdd HHMMSS'))] ;
elseif isfield(dout(1), 'SeriesDate') && isfield(dout(1), 'StudyTime')
    hdr_str = [hdr_str ,datestr(datenum([dout(1).SeriesDate,' ',dout(1).StudyTime],'yyyymmdd HHMMSS'))] ;
end

if isfield(dout(1), 'ImageComments')
   hdr_str = [hdr_str, ' ',dout(1).ImageComments] ;
end

disp(hdr_str)
disp(' ')

for ind = 1:length(ia)
    id = ia(ind) ;
    disp(['[',num2str(dout(id).SeriesNumber),']  ',datestr(datenum(dout(id).SeriesTime,'HHMMSS'),'HH:MM'), ...
        '  ',dout(id).SeriesDescription, ...
        '  ',dout(id).SequenceName])
    
    % entries taken from links provided in OsiriX to web site
    if isfield(dout,'SequenceVariant')
       sv = [dout(id).SequenceVariant] ;
       svc = textscan(sv,'%s','delimiter','\') ;
       svtxt = '';
       for isvc = 1:length(svc{:})
           svn = svc{:}{isvc} ;
           switch svn
               case 'SK'
                   txt = 'Segm ksp' ;
               case 'SS'
                   txt = 'Steady state' ;
               case 'TRSS'
                   txt = 'Time rev steady state' ;
               case 'SP'
                   txt = 'spoiled' ;
               case 'MP'
                   txt = 'mag prep';
               case 'OSP'
                   txt = 'oversamp phase';
               case 'MTC'
                   txt = 'mag trans contrast' ;
               otherwise
                   txt = '';
           end
           svtxt = [svtxt,' / ',txt];
       end
       disp(svtxt)
    end
    
    if isfield(dout,'ScanOptions')
       sv = [dout(id).ScanOptions] ;
       if ~isempty(sv)
       svc = textscan(sv,'%s','delimiter','\') ;
       svtxt = '';
       for isvc = 1:length(svc{:})
           svn = svc{:}{isvc} ;
           switch svn
               case 'PER'
                   txt = 'PE reordering' ;
               case 'RG'
                   txt = 'Resp gate' ;
               case 'CG'
                   txt = 'Cardiac gate' ;
               case 'PPG'
                   txt = 'Periph pulse gate' ;
               case 'FC'
                   txt = 'Flow Comp';
               case 'PFF'
                   txt = 'Partial Fourier Freq';
               case 'PFP'
                   txt = 'Partial Fourier Phase' ;
               case 'SP'
                   txt = 'Spatial presat' ;
               case 'FS'
                   txt = 'Fat sat' ;
               otherwise
                   txt = '';
           end
           svtxt = [svtxt,' / ',txt];
       end
       disp(svtxt)
       end
    end
    
    
    str = '' ;
    if isfield(dout,'VariableFlipAngleFlag')
        switch dout(id).VariableFlipAngleFlag
            case 'Y'
                str = ' (variable)' ;
        end
    end
    disp(['TE/TR/FA: ',num2str(dout(id).EchoTime),' / ', ...
        num2str(dout(id).RepetitionTime),' / ', ...
        num2str(dout(id).FlipAngle),str])
    
    k=strfind(dout(id).ScanningSequence,'IR') ;
    if ~isempty(k)
        disp(['Inv Time: ',num2str(dout(id).InversionTime)])
    end
    
    if isfield(dout,'AcquisitionMatrix')
        if dout(id).AcquisitionMatrix(1) == 0
            disp(['Aq matrix FEcol / PErow : ',num2str(dout(id).AcquisitionMatrix(2)), ...
                ' / ',num2str(dout(id).AcquisitionMatrix(3))])
        else
            disp(['Aq matrix FErow / PE col: ',num2str(dout(id).AcquisitionMatrix(1)), ...
                ' / ',num2str(dout(id).AcquisitionMatrix(4))])
        end
    end
    
    % following removed as superfluous given above Acq Matrix
    %if isfield(dout,'InPlanePhaseEncodingDirection')
    %    disp(['InPlanePE: ',dout(id).InPlanePhaseEncodingDirection])
    %end
    
    disp(['Sl thick/spacing: ',num2str(dout(id).SliceThickness), ...
        ' / ',num2str(dout(id).SpacingBetweenSlices)])
    
    disp(['Pixel Bandwidth ',num2str(dout(id).PixelBandwidth), ', nPE: ',...
        num2str(dout(id).NumberOfPhaseEncodingSteps), ', ETL: ', ...
        num2str(dout(id).EchoTrainLength), ...
        ', NAv: ',num2str(dout(id).NumberOfAverages) ])
    
    % this is to find total images in each series
    s_this = find(snall==dout(id).SeriesNumber) ;
    
    disp([num2str(dout(id).PixelSpacing(1)),'x',num2str(dout(id).PixelSpacing(2)), ' mm (', ...
        num2str(dout(id).Width),'x',num2str(dout(id).Height),'),  #ims: ', ...
        num2str(length(s_this))])
    
    %dout(ia(ind))
    disp(' ')
end

    
    






