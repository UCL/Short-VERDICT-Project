function [imslide, op] = swi2D(Z, varargin)
% SWI2D SWI style processing. 
% Complex image is divided by low-pass version. Phase is negated and scaled
% to form mask. Note power here is much higher than in literature.
% 
%  [imslide, op ] = swi2D(Z)
%  [imslide, op ] = swi2D(Z , param, value, ...)
%  where Z is complex data
%
% Param / Value Pairs
%   'swip' [5]  power to raise phase
%   'fw'   [0.25 0.25] filter width
%   'zthresh' [0] threshold for noise. Prevents "noise" from 
%             appearing, was set at about -0.02 when fw is [ 0.25 0.25]
%   'nsgp'  [5] number of slices grouped (should be an odd number). 7 would
%    form the minimum intensity projection from 3 before, the slice itself, and 3 after
%
%
% Outputs
%  op is a structure with fields
%   imswi, pmask
%   fw, zthresh, nsgp, swip
%   comment
%
% Based on Soman et al AJNR http://dx.doi.org/10.3174/ajnr.A3595
%   and WangJMRI12p661
%
% Example for Philips T2 FFE with phase saved. Also works for 3D SWI.
%
%  dffe = datparse ;
%  [vffe,mffe, locffe] = d2mat(dffe,{'slice','itype'},'op','dv') ;
%  cdata = vffe(:,:,:,1) .* exp(1i.* vffe(:,:,:,2)/1000) ; 
%  imswi = swi2D(cdata) ;
%  eshow(cat(2,cdata,imswi))
%  [mslide, mslab] = minip(imswi,7) ; % groups 7 slices
%  writeDicom(abs(mslide),'positive','header',{dffe locffe},'geom',mffe.geom,'burn_text','Non-Diagnostic')
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%  See also MINIP

swip = 5 ;
fw = [ 0.25 0.25 ] ;
zthresh = 0 ; 
nsgp = 5 ;

for ip = 1:2:length(varargin)
    switch varargin{ip}
        case 'swip'
            swip = varargin{ip+1} ;
        case 'fw'
            fw = varargin{ip+1} ;
        case 'zthresh'
            zthresh = varargin{ip+1} ;
        case 'nsgp'
            nsgp = varargin{ip+1} ;
        otherwise
            warning(['Unknown parameter: ',varargin{ip}])
    end
end


[ny nx nz] = size(Z) ;
imswi = Z ; % preallocate complex array
% pu = zeros(size(Z)) ;
% msk = zeros(size(Z)) ;

pmask = zeros([ny nx nz]) ;

for islice = 1:nz
    Zs = Z(:,:,islice) ;
    
    kZ =  i2k(Zs) ; 
    kZf = lpfilt2(kZ,[ny nx],fw) ;

    iZf = k2i(kZf) ; 
    
    %eshow(angle(Zs./iZf))
    hpphase = -angle(Zs./iZf) ; % NEGATE PHASE, then phi_v in Wang paper
    % pu(:,:,islice) = hpphase ;
    
    swimask = ones(size(Zs)) ;
    locnegp = find(hpphase < zthresh) ;
    
    hpphase_rs = hpphase./pi + 1 ; % -pi to 0 becomes  0 to 1
    hpphase_rs = hpphase_rs .^ swip ;
    
    swimask(locnegp) = hpphase_rs(locnegp);
    pmask(:,:,islice) = swimask ;
    
    % msk(:,:,islice) = swimask ;
    imswi(:,:,islice) = Zs.*swimask ;
    
end   
    
% mipgui(abs(mask(:,:,2:end-1).*ifZ(:,:,2:end-1)))

imslide = minip(imswi, nsgp) ;

op.imswi = imswi ; op.pmask = pmask ; op.fw =fw ; op.zthresh = zthresh ;
op.nsgp = nsgp ; op.swip = swip ;

op.comment = ['mIP slcs ',num2str(nsgp), ...
              ', filt [',num2str(fw(1)),' ',num2str(fw(2)),'], ', ...
              'power ',num2str(swip)] ;

end

function kout = lpfilt2(kin, FOV, w)
% Low-pass filter in 2D (Gaussian)
% kout = lpfilt2(kin, FOV, w)
% 

[ny nx] = size(kin) ;

kxc = ([1:nx]-(1+floor(nx/2)))/FOV(2) ;
kyc = ([1:ny]-(1+floor(ny/2)))/FOV(1) ;

[X,Y] = meshgrid(kxc, kyc) ;
filt = single(exp(-((X/w(2)).^2 + (Y/w(1)).^2 ))) ;

kout = filt .* kin;
end





    