function dinfo = cast_private(dinfo)
% CAST_PRIVATE Casts Private fields for cases where PACS or other nodes
% have changed the type.
% Some DICOMs from PACs seem to have some Private Fields with the correct
% Value Representation, but missing in others. 
% A symptom of a missing VR is that the MATLAB type will be uint8. For the
% Private Fields here, we would not be expecting uint8 and so this is used
% as the test. Note a more complete solution would be to use a proper Dicom
% dictionary that includes the Private Fields.
%
% dinfo = cast_private(dinfo)
%
% Copyright, 2019, David Atkinson.
% D.Atkinson@ucl.ac.uk
%
% See also dicom-dict-P1.txt  DATA_PROCESS
%
nf = length(dinfo) ;

dinfo = cp(dinfo,'Private_2001_1023','DS') ; % flip angle 
dinfo = cp(dinfo,'Private_2005_1030','FL') ; % Repetition Time
dinfo = cp(dinfo,'Private_2005_100e','FL') ; % MR ScaleSlope
dinfo = cp(dinfo,'Private_2005_1409','DS') ; % Rescale
dinfo = cp(dinfo,'Private_2005_140a','DS') ;
dinfo = cp(dinfo,'FlipAngle') ;  % Can  be set from Private Fields
                                 % to prevent rounding error
dinfo = cp(dinfo,'RepetitionTime') ; % Can be set from Private Field to
                     % avoid rounding to zero problem for subsecond TRs


end

function dinfo = cp(dinfo,field, VR)
if nargin<3
    VR='FL';
end


nf = length(dinfo) ;
ncast = 0 ;

switch VR
    case 'FL'
        for jf = 1:nf
            if isfield(dinfo(jf),field)
                if isa(dinfo(jf).(field),'uint8')
                %switch length(dinfo(jf).(field))
                    %case {4,8} % copes with cases when there are two TRs (B1 maps)
                        typ = 'single' ;
                        dinfo(jf).(field) = typecast(dinfo(jf).(field),typ) ;
                        ncast = ncast + 1;
                    %case {1,2}
                        % do nothing
                    %otherwise
                        %error(['unknown conversion: field:',field,' length: ',num2str(length(dinfo(jf).(field)))])
                %end
                end
            end
        end
    case 'DS'
        for jf = 1:nf
            if isfield(dinfo(jf),field)
                if isa(dinfo(jf).(field),'uint8')
                    str = char(dinfo(jf).(field)) ;
                    dinfo(jf).(field) = str2num(str') ;
                    ncast = ncast + 1;
                end
            end
        end
    otherwise
        error(['Unkonwn VR'])
end

if ncast > 0
    disp([' Cast ',num2str(ncast),' entries for field: ',field])
end
end


