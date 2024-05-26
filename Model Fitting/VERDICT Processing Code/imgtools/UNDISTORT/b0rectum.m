function b0 = b0rectum(ir, varargin)
% B0RECTUM

offset = 0 ;
shift = 0 ;
amp = 100 ;
low_signal_loc = [] ;
irectA = 0 ;
irectP = 0 ;

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    switch varargin{ipv}
        case 'offset'
            offset = val ;
        case 'shift'
            shift = val ;
        case 'low_signal_loc'
            low_signal_loc = val ;
        case 'amp'
            amp = val ;
        case 'irectA'
            irectA = val ;
        case 'irectP'
            irectP = val ;
        otherwise
            error('Unknown parameter')
    end
end

edgeA = irectA - shift ;
edgeP = irectP - shift ;

b0 = offset + amp*(1 -( 1/(1 + exp(-(ir-edgeA)/3)) + 1/(1 + exp((ir-edgeP)/3)))) ;

% b0 = offset + amp*(1 -( 1/(1 + exp(-(r-28+shift)/6)) + 1/(1 + exp((r-67+shift)/6)))) ;

if ismember(ir, low_signal_loc)
    b0 = 0 ;
end

end