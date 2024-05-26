function cfa = D2CFA(D)
% D2CFA Diffusion tensor to colour coded FA map
%
% cfa = D2CFA(D)
%
% D   is [ny nx nz 3 3]
% cfa is [ny nx nz 3]
%
% $Id: d2cfa.m 268 2010-01-08 17:33:22Z ucacdat $ David Atkinson
%
% See also tshow_comp_image

locOK = isfinite(D) ;
notOK = find(locOK==0) ;
D(notOK) = 0 ;

fa = invariantsb(D,'fa') ;

vec = zeros(size(D,1), size(D,2), size(D,3), 3) ;
for id1 = 1:size(D,1)
    for id2 = 1:size(D,2)
        for id3 = 1:size(D,3)
            [U,S,V] = svd(squeeze(D(id1,id2,id3,:,:)),0);
            % U should be in LPS (DICOM)
            % want L red, P green, S blue ie no swaps needed
            vec(id1,id2,id3,:) = reshape(U(:,1),[1 1 1 3]);
        end
    end
end

fa(find(fa>1))=1 ;

cfa = vec2col(vec(:,:,:,[1 2 3]),fa,0,1) ;
