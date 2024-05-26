function varargout = stejskal(delta, DELTA, varargin)
% STEJSKAL Stejskal-Tanner equation to return either b-value or gradient strength
%
% bval = stejskal(delta, DELTA, G=Gvalue)
% G    = stejskal(delta, DELTA, bval=bvalue)
%
% delta , DELTA [ms]
% G [mT/m] 
% bval [s/mm^2]
%
% Example
%  stejskal(23.9, 43.8, G=32)       gives b-value of 1500
%  stejskal(23.9, 43.8, bval=1500)  gives G of 32
%
% Copyright 2022. David Atkinson
%
% See also verdict_fit

gamma_ms = 2.6752218744 * 1e+05  ; % 2.pi.gamma [using ms not s]

switch varargin{1}
    case "G"
        G = varargin{2} ;

        GTpm = G * 1e-3 ;

        b = gamma_ms^2 * delta^2 * GTpm^2 * (DELTA - delta/3) ;

        bval = b * 1e-3 * 1e-6 ; % ms / m^2  to s/mm^2

        varargout{1} = bval ;

    case "bval"
        b = varargin{2} / 1e-9 ;
        GTpm = sqrt(b/(gamma_ms^2 * delta^2 *(DELTA - delta/3))) ;
        G = GTpm/1e-3 ;
        varargout{1} = G ;

    otherwise
        error('MATLAB:stejskal:unknownInput', ...
            ['Unrecognised input: ', varargin{1}])
end
