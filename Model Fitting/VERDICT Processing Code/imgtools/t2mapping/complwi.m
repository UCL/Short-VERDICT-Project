function complwi

fstem = '/Users/davidatkinson/University College London/Gong, Yangcan - Luminal Water Imaging' ;

dT2 = datparse(dselect('message','Select T2')) ;
dME = datparse(dselect('message','Select multi-echo')) ;
% dLWI15 = datparse(dselect('message','Select 1.5mm LWI')) ; 
% dLWI09 = datparse(dselect('message','Select 0.9mm LWI')) ; 
db = datparse(dselect('message','Select b2000')) ; 


[vT2, mT2] = d2mat(dT2,{'slice'},'op','fp') ;
[vME, mME] = d2mat(dME, {'slice','echo'},'op','fp') ;

bilwiME = bifit(vME,mME.effTEVec_indata / 1000) ;

last3 = sum(vME(:,:,:,6:8),4) ;

eshow(bilwiME)
eshow(last3) 

% [v15, m15] = d2mat(dLWI15, {'slice','echo'},'op','fp') ;
% [v09, m09] = d2mat(dLWI09, {'slice','echo'},'op','fp') ;
[vb, mb] = d2mat(db, {'slice'},'op','fp') ;

eshow(vT2)
T2max = input('Enter max T2 for display') ;
ylims = input('Enter row min max');
xlims = input('Enter col min max') ;
zlims = input('Enter slice min max') ;

eshow(vb)
blims = input('Enter min and max b for display') ;



% bilwi15 = bifit(v15,m15.effTEVec_indata / 1000) ;
% bilwi09 = bifit(v09,m09.effTEVec_indata / 1000) ;
% 
% v15rs = dreslice(bilwi15, m15, mT2) ;
% v09rs = dreslice(bilwi09, m09, mT2) ;

bilwiME = bifit(vME,mME.effTEVec_indata / 1000) ;
vbrs  = dreslice(vb, mb, mT2) ;


vT2 =  mat2gray(vT2,[0 T2max]) ;
vbrs = mat2gray(vbrs, [blims(1) blims(2)]) ;
v15rs = mat2gray(v15rs, [0 0.4]) ;
v09rs = mat2gray(v09rs,[0 0.4]) ;

img = cat(1, vT2(ylims(1):ylims(2),xlims(1):xlims(2),zlims(1):zlims(2)), ...
    vbrs(ylims(1):ylims(2),xlims(1):xlims(2),zlims(1):zlims(2)), ...
    v15rs(ylims(1):ylims(2),xlims(1):xlims(2),zlims(1):zlims(2)) ) ;
 % ...
    %v09rs(ylims(1):ylims(2),xlims(1):xlims(2),zlims(1):zlims(2)) ) ;

eshow(img)

return



j = 1 ;

% Reimagine
d{j}.F15lwf = fullfile(fstem,'Reimagine','2010034', '1p5mm', 'LI_greyscale' ) ;
d{j}.F15dat = fullfile(fstem,'Reimagine','2010034', '1p5mm', 'I17') ;

d{j}.F09lwf = fullfile(fstem,'Reimagine','2010034', '0p9mm', 'LI-greyscale' ) ;
d{j}.F09dat = fullfile(fstem,'Reimagine','2010034', '0p9mm', 'I15') ;

j = 2 ;
d{j}.F15lwf = fullfile(fstem,'Reimagine','2010026', '1p5mm', 'LI_greyscale' ) ;
d{j}.F15dat = fullfile(fstem,'Reimagine','2010026', '1p5mm', 'I20') ;

d{j}.F09lwf = fullfile(fstem,'Reimagine','2010026', '0p9mm', 'LI_greyscale' ) ;
d{j}.F09dat = fullfile(fstem,'Reimagine','2010026', '0p9mm', 'I18') ;

j=3 ;
d{j}.F15lwf = fullfile(fstem,'Reimagine','2010020', '1p5mm', 'LI_greyscale' ) ;
d{j}.F15dat = fullfile(fstem,'Reimagine','2010020', '1p5mm', 'I18') ;

d{j}.F09lwf = fullfile(fstem,'Reimagine','2010020', '0p9mm', 'LI_greyscale' ) ;
d{j}.F09dat = fullfile(fstem,'Reimagine','2010020', '0p9mm', 'I16') ;



dinfoF15lwf = dparse(d{j}.F15lwf) ;
[vF15lwf, m15] = d2mat(dinfoF15lwf, {'slice'},'op','dv') ;

dinfoF15dat = datparse(d{j}.F15dat) ;
[vF15dat, m15d] = d2mat(dinfoF15dat,{'slice','echo'},'op','fp') ;
bilw15 = bifit(vF15dat,m15d.effTEVec_indata / 1000) ;

dinfoF09lwf = dparse(d{j}.F09lwf) ;
[vF09lwf, m09d] = d2mat(dinfoF09lwf , {'slice'},'op','dv') ;

dinfoF09dat = datparse(d{j}.F09dat) ;
[vF09dat, m09d] = d2mat(dinfoF09dat,{'slice','echo'},'op','fp') ;
bilw09 = bifit(vF09dat,m09d.effTEVec_indata / 1000) ;

eshow(cat(2,vF15lwf/100, bilw15), 'Name','F bi 15')
eshow(cat(2,vF09lwf/100, bilw09), 'Name','F bi 09')

end

function lwf = bifit(v,TEs)
% Fixed T2 fit.

T2short  =  50e-3 ; %  short T2 
T2long   = 300e-3 ; %  long T2

data = double(v) ;
szdata = size(data) ;
necho = szdata(end) ;        % number of echoes
np = prod(szdata(1:end-1)) ; % number of pixels in data
rdata = reshape(data, [np necho]) ; % reshape to make looping and parfor easier

sigs = rdata' ;
A = [exp(-TEs(:)/T2short)  exp(-TEs(:)/T2long) ] ;

ac = A\sigs ;

lwf = ac(2,:)./ (ac(1,:) + ac(2,:)) ;

lwf = reshape(lwf,szdata(1:end-1)) ;

lwf(lwf<0)=0;
lwf(lwf>1)=1;

end

