function explore_colourmaps
% EXPLORE COLOURMAPS and Fusion / Overlays
%

% Make explanatory figure

nrow = 30 ;
nscalar = 256 ;
scalars = linspace(0,1, nscalar) ;

imrow = scalars ;
imScalar = repmat(imrow,[nrow 1]) ;

hf = figure ;
t = tiledlayout(6,1,'TileSpacing','none','Padding','tight') ;
nexttile
ha=imshow(imScalar,[0 1]);
ha.Parent.XAxisLocation = 'top' ;
ha.Parent.Visible = "on" ;
ha.Parent.YTick=[] ;
ha.Parent.YMinorTick="off" ;

scTick = [0 0.2 0.4 0.6 0.8 1.0] ;
ha.Parent.XTick = 0.5 + 256.*scTick ;
for iTick = 1:length(scTick)
        XLab{iTick} = num2str(scTick(iTick)) ;
end
ha.Parent.XTickLabel = XLab ;


nexttile(5)
y = 0.7 + 0.05*sin(2*pi*scalars/(0.05)) ;
imBase = repmat(y,[nrow 1]) ;

imshow(imBase,[0 1])

nexttile(6)
scaleBase=[1 1 1] ;
[imRGB, imAlphaRGB, imCmapRGB, imScalarRemap, alpha, opts] = blendImScalar(imBase, imScalar, scaleBase) ;

imshow(imRGB)

nexttile(4)
imshow(imAlphaRGB)

nexttile(2)
imshow(imCmapRGB)

nexttile(3)
plot(imrow,imScalarRemap(1,:),'r')
hold on
plot(imrow,alpha(1,:),'b')

end



