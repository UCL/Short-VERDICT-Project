function [dradial, daxial] = d2radial(D)
% D2RADIAL Axial and radial (parallel and perpendicular) diffusivities
%
% [dradial, daxial] = d2radial(D)
%
% D.Atkinson@ucl.ac.uk


dradial = zeros(size(D,1),size(D,2),size(D,3)) ;
daxial = zeros(size(D,1),size(D,2),size(D,3)) ;
for id1 = 1:size(D,1)
    for id2 = 1:size(D,2)
        for id3 = 1:size(D,3)
            [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
            if sum(eval) > realmin && isfinite(sum(eval))
                dradial(id1,id2,id3) = (eval(2)+eval(3)) /2;
                daxial(id1,id2,id3) = eval(1);
            else
                dradial(id1,id2,id3) = 0 ;
                daxial(id1,id2,id3) = 0 ;
            end
        end
    end
end