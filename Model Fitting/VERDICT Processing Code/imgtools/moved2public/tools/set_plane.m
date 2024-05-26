function plane = set_plane(varargin)
% SET_PLANE Set a plane structure
%
% plane = set_plane('pn',P,N)
% plane = set_plane('pvv',P1,V1,V2)
% plane = set_plane('ppp',P1,P2,P3)
%
%
% plane returned as a structure with fields:
%   .norm
%   .point
%   .abcd   and .a, .b, .c, .d
%
% see MathWorld plane
% David Atkinson, UCL.  D.Atkinson@ucl.ac.uk
%
% See also point_plane

dpt = 0.9 ; % threshold, above which vectors considered to be colinear

switch varargin{1}
    case {'ppp','PPP'}
        P1 = varargin{2} ;
        P2 = varargin{3} ;
        P3 = varargin{4} ;
        
        V1 = P2 - P1 ;
        V2 = P3 - P1 ;
        
        plane = set_plane('pvv',P1,V1,V2) ;
    case{ 'pvv', 'PVV' }
        P1 = varargin{2} ;
        V1 = varargin{3} ;
        V2 = varargin{4} ;
        
        
        V1 = V1 ./norm(V1) ;
        V2 = V2 ./norm(V2) ;
        
        if abs(sum(V1.*V2)) > dpt
            warning(['set_plane: Almost co-linear!'])
        end
        
        N = cross(V1,V2) ;
        N = N ./ norm(N) ;
        
        plane.normal = N ;
        plane.point  = P1 ;
        
        plane = set_abcd(plane) ;
        
    case {'pn','PN'}
        P = varargin{2} ;
        N = varargin{3} ;
        N = N./norm(N)  ;
        
        plane.normal = N ;
        plane.point = P  ;
        
        plane = set_abcd(plane) ;
    otherwise
        disp(['Not implemented'])
end

%-----------------------------------------------

function plane = set_abcd(plane)
% SET_ABCD
%
% 

N = plane.normal ;
N = N./norm(N) ;
P = plane.point ;

plane.d = -dot(N,P) ;
plane.a = N(1) ;
plane.b = N(2) ;
plane.c = N(3) ;

plane.abcd = [plane.a plane.b plane.c plane.d] ;

