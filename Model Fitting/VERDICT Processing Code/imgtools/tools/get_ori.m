function varargout = get_ori(vec, varargin)
% GET_ORI Get orientation. SUPERCEDED BY ORI_INFO
% Finds radiological plane with normal closest to suppled vector, or cross
% product if 6-element IOP array input. 
%
%  ori = get_ori(vec_lph) 
%  ori = get_ori(iop) 
%  ori = get_ori(..., param, value, ...)
%
%    ori is 1 for TRA, 2 for SAG and 3 for COR
%  [ori, dstr] = ...
%     dstr will be 'LR' or 'RL', etc
%  [ori, dstr, oristr] = ...
%
% param
% 'confirm_radiological' [false]
%
% D.Atkinson@ucl.ac.uk
%

confirm_radiological = false ;

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    switch varargin{ipv}
        case 'confirm_radiological'
            confirm_radiological = val ;
        otherwise
            error(['Unknown input: ',varargin{ipv}]) ;
    end    
end
                
if length(vec) == 6
    sdc = cross(vec(1:3), vec(4:6)) ;
elseif length(vec) == 3
    sdc = vec ;
else
    error(['First arg must be normal vector or IOP vectors'])
end

if abs(norm(sdc)-1) > 0.01
    warning(['Expecting vector norm of 1, found: ', num2str(norm(sdc)),' Normalising'])
    sdc = sdc ./ norm(sdc) ;
end

sagnorm = cross([0 1 0],[0 0 -1]) ; % LPH coordinate
cornorm = cross([1 0 0],[0 0 -1]) ;
tranorm = cross([1 0 0],[0 1  0]) ;

sagproj = dot(sagnorm, sdc) ; 
corproj = dot(cornorm, sdc) ;
traproj = dot(tranorm, sdc) ;

projs = [traproj sagproj corproj ] ;
oristr  = {'TRA', 'SAG', 'COR'} ; % correct order?
    
[~, iproj] = max(abs(projs));

if projs(iproj) < 0 && confirm_radiological
    warning(['Dot product is negative - not radiological orientation'])
end

vds = [ 1 0 0 ; -1 0 0 ; 0 1 0 ; 0 -1 0 ; 0 0 1 ; 0 0 -1] ;
dstr = { 'RL' , 'LR'   , 'AP' , 'PA' , 'FH' , 'HF' } ;

dps = dot(repmat(sdc(:)',[6 1]), vds, 2) ;
[mdp, imdp] = max(dps) ;

disp(['Vec is ',dstr{imdp},'. Plane norm with closest orientation is: ',...
       oristr{iproj}, ' having dot product: ', num2str(projs(iproj))])
   
if nargout > 0
   varargout{1} = iproj ;
end

if nargout > 1
    varargout{2} = dstr{imdp} ;
end

if nargout > 2
    varargout{3} = oristr{iproj} ;
end



