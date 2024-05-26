function gibbs


nhr = 1024;
nlr = 128 ;
imghr = prostate(nhr) ;
imglr = prostate(nlr) ;

xinhr = imghr(:,477) ;
xinlr = imglr(:,60) ;

khr = i2k(xinhr) ;

mask = zeros([nhr 1]) ;
dc = sz2DC(nhr) ;
mask(dc-64:dc+63) = 1 ;

khr_masked = khr .* mask ;

figure
plot(xinhr), hold on
plot(abs(k2i(khr_masked)))

ilr = k2i(khr(mask==1)) ;
figure
plot(abs(ilr))

klr = i2k(xinlr) ;
ph = ([1:nlr]'-sz2DC(nlr) ) *pi/nlr ;

plot(abs(i2k(exp(1i*ph) .* klr)))

figure
for shft = [0:0.1:20]
    ph = ([1:nlr]'-sz2DC(nlr) ) * shft * pi/nlr ;
    plot(abs(i2k(exp(1i*ph) .* klr)))
    axis([0 130 0 100])
    title(['Shift ',num2str(shft)])
    drawnow
end
