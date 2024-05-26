function [t2s, t2sd] = de2t2s(vol, mparams)
% DE2T2S Dual echo to T2*
% [t2s, t2sd] = de2t2s(vol, mparams)

nbase = 15 ;
slc = 3 ;


[ny nx nz ndyn ne] = size(vol) ;

deltaTE = ( mparams.effTEVec_indata(2) -  mparams.effTEVec_indata(1) ) ;

if ndyn < nbase
    error(['Insufficient dynamics: ',num2str(ndyn)])
end

ve1 = sum(vol(:,:,:,1:nbase,1),4) / nbase ;
ve2 = sum(vol(:,:,:,1:nbase,2),4) / nbase ;


% S = S0 exp(-TE/T2S)
%
% Se1/Se2 = exp(-TE1/T2s) / exp(-TE2/T2S)
%
% ln(Se1/Se2) = -TE1/T2S -- TE2/T2S = 1/T2S ( TE2 - TE1) 
%
% T2S = (TE2 - TE1) / ln(Se1/Se2)

t2s = deltaTE / log(ve1./ve2) ;

eshow(t2s)

t2sd = deltaTE ./ log(vol(:,:,slc,:,1)./vol(:,:,slc,:,2)) ;

eshow(t2sd)


