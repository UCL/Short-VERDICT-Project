function mrecon2dicom(MR, varargin)
% MRECON2DICOM MRecon to Dicom NEW!! SEE MR2geom
%
% Assumes IPP is top left CENTRE
%
% mrecon2dicom(MR, parameter, value, ...)
%
% Parameter:
%  'slice'  default central slice
%
% Currently returns for one slice
%
%
% D.Atkinson@ucl.ac.uk
%

% Add check that MR is in image domain

if MR.Parameter.Scan.Stacks ~= 1
    warning(['Only for 1 stack'])
end

nz = size(MR.Data,3) ;
islice = ceil(nz/2) ; % default slice

for ipv = 1:2:length(varargin)
    switch varargin{ipv}
        case 'slice'
            islice = varargin{ipv+1} ;
            if islice < 1 || islice > nz
                warning(['slice out of data: ',num2str(islice)])
            end
        otherwise
            warning(['Unknown parameter: ',varargin{ipv}])
    end
end

IPP  = MR.Transform( [1;1;islice], 'ijk', 'RAF' )'  % RAF is LPH !
IOPW = MR.Transform( [1;2;islice], 'ijk', 'RAF')' - IPP ;
IOPH = MR.Transform( [2;1;islice], 'ijk', 'RAF')' - IPP ;


PS_HW = [norm(IOPH) norm(IOPW)]

IOPW/PS_HW(2)
IOPH/PS_HW(1)




