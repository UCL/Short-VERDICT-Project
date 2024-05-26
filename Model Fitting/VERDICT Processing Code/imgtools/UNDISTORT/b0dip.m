function b0 = b0dip(r, varargin)
% B0DIP

offset = 0 ;
shift = 0 ;
amp = 80 ;
zero_rectum = false ;
low_signal_loc = [] ;

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    switch varargin{ipv}
        case 'offset'
            offset = val ;
        case 'shift'
            shift = val ;
        case 'zero_rectum'
            zero_rectum = val ;
        case 'low_signal_loc'
            low_signal_loc = val ;
        case 'amp'
            amp = val ;
    end
end

b0 = offset + amp*(1 -( 1/(1 + exp(-(r-28+shift)/6)) + 1/(1 + exp((r-67+shift)/6)))) ;
% b0 = offset + 80*(1 -( 1/(1 + exp(-(r-10)/5)) + 1/(1 + exp((r-40)/5)))) ;
if zero_rectum
    if r>32 && r < 51
        b0=0;
    end
end

b0(low_signal_loc) = 0 ;

end