function ft = run_CVNNLS(data, techo)
% !!! OLD CODE  See T2FITF
%
% Code copied from t2pdfit - all needs tidying up

data = double(abs(data)) ; % convert to modulus double in case complex or single
szdata = size(data) ;
szmap = szdata(1:end-1) ; % Maps are same size as image component of data.

necho = szdata(end) ;        % number of echoes
np = prod(szdata(1:end-1)) ; % number of pixels in data
rdata = reshape(data, [np necho]) ; % reshape to make looping and parfor easier

% Default mask
mask = ones([np 1]) ;

loc_mask = find(mask) ;
nloc = length(loc_mask) ;

mdata = rdata(loc_mask,:) ;  % just the pixels in the mask (helps parfor)

if length(techo) > 1
    TEs = techo ;
else
    TEs = [1:necho].*techo ;
end

if TEs(1) > 1
    warning(['Expecting TEs in s, dividing by 1000'])
    TEs = TEs ./ 1000 ;
end
 
% from CVNNLS.m
T2min = TEs(1)*1.5;
T2max = 3*TEs(end);
T2length = 120;
T2Times = logspace( log10( T2min ), log10( T2max ), T2length );
locT2short = find(T2Times < 200e-3) ;
locT2long = find(T2Times >= 200e-3) ;

A = exp( -kron( TEs',1./T2Times ) );
flwi = cell(nloc, 1);

parfor iloc = 1: nloc
    [ xOut, ChiUsed, resid ] = CVNNLS(A,mdata(iloc,:)') ;
    
    A1 = trapz( T2Times(locT2short), xOut(locT2short) ) ;
    A2 = trapz( T2Times(locT2long),  xOut(locT2long) ) ;
    LWF = A2 ./ (A1+A2) ;
    
    flwi{iloc} = struct('LWF',LWF, 'xOut',xOut) ;
    
end

ft.A = A ;
ft.T2Times = T2Times ;
ft.flwi = flwi ;
ft.szmap = szmap ;


ft.nloc = nloc ;
ft.loc = loc_mask ;

