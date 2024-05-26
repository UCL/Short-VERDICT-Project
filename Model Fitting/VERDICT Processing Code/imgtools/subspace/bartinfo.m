function bartinfo(varargin)
% BARTINFO 
%
% bartinfo('traj', traj)
%
% 

npv = length(varargin) ;

for ipv = 1:2:npv
    param = varargin{ipv} ;
    val   = varargin{ipv+1} ;
    varn  = inputname(ipv+1) ;
    
    switch param
        case 'traj'
            traj = val ;
            
            sztraj = size(traj) ;
            
            if sztraj(1)~=3
                error(['traj expects size 3 in first dimension.'])
            end
            
            kc  = reshape(traj,[3 prod(sztraj(2:end))]) ;
            
            figure('Name', varn)
            plot(kc(1,:), kc(2,:),'.')
            axis equal
            
        otherwise
            error(['Unknown param: ',param])
    end
    
    
end

