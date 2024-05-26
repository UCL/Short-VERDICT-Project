function [T2, figname, vet2, met2, S0] = MET2(fn_t2)
% MET2 Multi-echo T2 fitting
%  [T2, figname, vet2, met2, S0] = MET2
%  [T2, figname, vet2, met2, S0] = MET2(fn_t2)
%
%   fn_t2  is a cell array
%
% Calls dselect for user to enter filename of multi-echo, single slice file
% Performs a linear fit to the log of the non-zero signal for
%   all pixels where the first echo is above a threshold
%   fit does not use the first echo
%
%
% Use T1display to plot, e.g.
%
%  T1display(T2)  (will label with variable name, not "T1")
%
%
% D.Atkinson@ucl.ac.uk
%
% See also T1display, datparse, dselect


start_echo = 2  % first echo used in fit
thresh = 15 ;   % threshold for pixels to fit

if nargin>0 && exist(fn_t2{1},'file')
    dt2 = datparse(fn_t2) ;
else
    dt2 = datparse(dselect('message','Select ME T2')) ;
end

% Expecting single slice, multiple echoes
ityps = [dt2.itype] ;
if ~isempty(find(ityps == 7, 1))
    [vt2map,mt2map] = d2mat(dt2,{'slice','itype'},'itype',7,'op','fp') ;
    eshow(vt2map,'Name','Scanner calculated T2 map')
else
    warning(['No scanner calculated T2 map.'])
end

% [vet2,met2] = d2mat(dt2,{'effTE','slice','itype'}, ...
%     'effTE',mt2map.effTEVec_indata(2:end),'itype',11,'op','fp') ;

[vet2,met2, locs] = d2mat(dt2,{'slice','echo','itype'}, 'itype',11,'op','fp') ;
met2.effTE = unique([dt2(locs).EffectiveEchoTime]) ;

[ny, nx, ns, nt] = size(vet2) ;
e1 = vet2(:,:,1,1) ;
loc = find(e1 > max(e1(:))/thresh ) ;
Mv = reshape(vet2(:,:,:,start_echo:nt), [nx*ny*ns (nt-start_echo+1)]) ;

nloc = length(loc) ;
T2_par = zeros(nloc,1) ;
S0_par = zeros(nloc,1) ;

T2 = zeros([ny nx]) ;
S0 = zeros([ny nx]) ;

x = met2.effTE(start_echo:end)' ;

Mvle = double(Mv(loc,:)) ; % prevents a broadcast variable

parfor iloc = 1:nloc
    % General model Exp1:
    % f(x) = a*exp(b*x)
        y = double(Mvle(iloc,:)) ;
        
        fitobj = fit(x(:),y(:),'exp1') ;
        T2_par(iloc) = -1/fitobj.b ;
        S0_par(iloc) = fitobj.a ;
end
T2(loc) = T2_par ;
S0(loc) = S0_par ;

ffn = dt2.Filename ;
[~,fn,ext] = fileparts(ffn) ;
figname = [fn, ext] ;

eshow(T2,'Name',['T2 map for ',figname])

if exist('vt2map','var')
    eshow(T2-vt2map, 'Name','Fitted - scanner map')
end


     





