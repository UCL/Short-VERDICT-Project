function kspout = upsense(ksp, sensef)

szin = size(ksp) ;

if szin(3) > 1 && szin(6) > 1
    error(['Cannot cope with 3D and more than one loca'])
end
if szin(3) > 1 && sensef(3) > 1
    error(['Cannot have SENSE in 3D if not 3D'])
end

if szin(6) > 1
    ksp = permute(ksp,[1 2 6 4 5 3 7:length(szin)]) ;
    perm = true ;
else 
    perm = false ;
end

szout = size(ksp) ; 

% loop through dims 1 to 3 and grow if SENSE factor, then 
% squeeze, do interpolation and reshape back.

ksp = squeeze(ksp) ; % interp cant handle singleton dims

% Do this crudely by interpolation rather than full sinc.
for idim = 1:ndims(ksp) 
    kcin = [1:size(ksp,idim)] - DCgp(ksp,idim);
    
    kcout = kcin ;
    
    if idim < 4
        kcin = kcin * sensef(idim) ;
        nout = round(length(kcin) * sensef(idim)) ;
        kcout = [1:nout] - sz2DC(nout) ;
        szout(idim) = nout ;
    end
    gridVecsin{idim} = kcin ;
    gridVecsout{idim} = kcout ;
end


F = griddedInterpolant(gridVecsin, ksp) ;

kspout = F(gridVecsout) ;

kspout = reshape(kspout, szout) ;  

if perm
    kspout = permute(kspout,[1 2 6 4 5 3 7:length(szin)]) ;
end





